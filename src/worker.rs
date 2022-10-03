#![feature(int_roundings)]

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::msm::VariableBaseMSM;
use ark_ff::{FftField, Field, PrimeField, ToBytes, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Radix2EvaluationDomain, UVPolynomial,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use capnp::{capability::Promise, message::ReaderOptions};
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};

use hello_world::{
    config::NetworkConfig,
    hello_world_capnp::{plonk_peer, plonk_slave},
    utils::{deserialize, serialize, FftWorkload},
};
use rand::{rngs::ThreadRng, thread_rng};
use tokio::runtime::Handle;

use core::slice;
use futures::{future::join_all, AsyncReadExt};
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    net::{SocketAddr, ToSocketAddrs},
    rc::Rc,
    sync::{Arc, Mutex},
};

struct FftTask {
    is_quot: bool,
    is_inv: bool,
    is_coset: bool,

    workloads: Vec<FftWorkload>,
    rows: Vec<Vec<Fr>>,
    cols: Vec<Vec<Fr>>,
}

struct State {
    rng: ThreadRng,

    me: usize,
    network: NetworkConfig,

    bases: Vec<G1Affine>,
    domain: Radix2EvaluationDomain<Fr>,
    c_domain: Radix2EvaluationDomain<Fr>,
    r_domain: Radix2EvaluationDomain<Fr>,
    quot_domain: Radix2EvaluationDomain<Fr>,
    quot_c_domain: Radix2EvaluationDomain<Fr>,
    quot_r_domain: Radix2EvaluationDomain<Fr>,

    fft_tasks: HashMap<u64, FftTask>,

    wire: DensePolynomial<Fr>,
}

#[derive(Clone)]
struct PlonkImpl {
    state: Arc<State>,
}

fn fft1_helper(
    v: &mut Vec<Fr>,
    i: u64,
    is_coset: bool,
    is_inv: bool,
    domain: &Radix2EvaluationDomain<Fr>,
    c_domain: &Radix2EvaluationDomain<Fr>,
    r_domain: &Radix2EvaluationDomain<Fr>,
) {
    if is_coset && !is_inv {
        let g = Fr::multiplicative_generator();
        v.iter_mut()
            .enumerate()
            .for_each(|(j, u)| *u *= g.pow([i + j as u64 * r_domain.size]));
    }
    if is_inv {
        c_domain.ifft_in_place(v);
    } else {
        c_domain.fft_in_place(v);
    }
    let omega_shift = if is_inv {
        domain.group_gen_inv
    } else {
        domain.group_gen
    };
    v.iter_mut()
        .enumerate()
        .for_each(|(j, u)| *u *= omega_shift.pow([i * j as u64]));
}

fn fft2_helper(
    v: &mut Vec<Fr>,
    i: u64,
    is_coset: bool,
    is_inv: bool,
    c_domain: &Radix2EvaluationDomain<Fr>,
    r_domain: &Radix2EvaluationDomain<Fr>,
) {
    if is_inv {
        r_domain.ifft_in_place(v);
    } else {
        r_domain.fft_in_place(v);
    }
    if is_coset && is_inv {
        let g = Fr::multiplicative_generator().inverse().unwrap();
        v.iter_mut()
            .enumerate()
            .for_each(|(j, u)| *u *= g.pow([i + j as u64 * c_domain.size]));
    }
}

fn commit_polynomial(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    let mut scalars = scalars.iter().map(|s| s.into_repr()).collect::<Vec<_>>();

    scalars.resize(bases.len(), Fr::zero().into_repr());

    VariableBaseMSM::multi_scalar_mul(&bases, &scalars)
}

impl plonk_slave::Server for PlonkImpl {
    fn init(
        &mut self,
        params: plonk_slave::InitParams,
        _: plonk_slave::InitResults,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();
        let domain_size = p.get_domain_size() as usize;
        let quot_domain_size = p.get_quot_domain_size() as usize;

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let mut bases = vec![];
        p.get_bases()
            .unwrap()
            .iter()
            .for_each(|n| bases.extend_from_slice(n.unwrap()));
        state.bases = deserialize(&bases).to_vec();
        {
            state.domain = Radix2EvaluationDomain::<Fr>::new(domain_size).unwrap();
            let r = 1 << (self.state.domain.log_size_of_group >> 1);
            let c = self.state.domain.size() / r;
            state.r_domain = Radix2EvaluationDomain::<Fr>::new(r).unwrap();
            state.c_domain = Radix2EvaluationDomain::<Fr>::new(c).unwrap();
        }
        {
            state.quot_domain = Radix2EvaluationDomain::<Fr>::new(quot_domain_size).unwrap();
            let r = 1 << (self.state.quot_domain.log_size_of_group >> 1);
            let c = self.state.quot_domain.size() / r;
            state.quot_r_domain = Radix2EvaluationDomain::<Fr>::new(r).unwrap();
            state.quot_c_domain = Radix2EvaluationDomain::<Fr>::new(c).unwrap();
        }
        Promise::ok(())
    }

    fn var_msm(
        &mut self,
        params: plonk_slave::VarMsmParams,
        mut results: plonk_slave::VarMsmResults,
    ) -> Promise<(), capnp::Error> {
        let mut scalars = vec![];
        let p = params.get().unwrap();

        let workload = p.get_workload().unwrap();

        let start = workload.get_start() as usize;
        let end = workload.get_end() as usize;

        p.get_scalars()
            .unwrap()
            .iter()
            .for_each(|n| scalars.extend_from_slice(deserialize(n.unwrap())));

        results
            .get()
            .set_result(serialize(&[VariableBaseMSM::multi_scalar_mul(
                &self.state.bases[start..end],
                &scalars,
            )]));

        Promise::ok(())
    }

    fn fft_init(
        &mut self,
        params: plonk_slave::FftInitParams,
        _: plonk_slave::FftInitResults,
    ) -> Promise<(), capnp::Error> {
        let params = params.get().unwrap();

        let id = params.get_id();
        let workloads = params
            .get_workloads()
            .unwrap()
            .into_iter()
            .map(|i| FftWorkload {
                row_start: i.get_row_start() as usize,
                row_end: i.get_row_end() as usize,
                col_start: i.get_col_start() as usize,
                col_end: i.get_col_end() as usize,
            })
            .collect::<Vec<_>>();
        let is_quot = params.get_is_quot();
        let is_inv = params.get_is_inv();
        let is_coset = params.get_is_coset();

        let r_domain = if is_quot {
            self.state.quot_r_domain
        } else {
            self.state.r_domain
        };

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        state.fft_tasks.insert(
            id,
            FftTask {
                is_coset,
                is_inv,
                is_quot,
                rows: vec![vec![]; workloads[self.state.me].num_rows()],
                cols: vec![
                    vec![Fr::default(); r_domain.size()];
                    workloads[self.state.me].num_cols()
                ],
                workloads,
            },
        );

        Promise::ok(())
    }

    fn fft1(
        &mut self,
        params: plonk_slave::Fft1Params,
        _: plonk_slave::Fft1Results,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();

        let mut v = vec![];

        p.get_v()
            .unwrap()
            .iter()
            .for_each(|n| v.extend_from_slice(deserialize(n.unwrap())));

        let i = p.get_i();
        let id = p.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_tasks.get_mut(&id).unwrap();

        let (domain, c_domain, r_domain) = if task.is_quot {
            (
                self.state.quot_domain,
                self.state.quot_c_domain,
                self.state.quot_r_domain,
            )
        } else {
            (self.state.domain, self.state.c_domain, self.state.r_domain)
        };

        fft1_helper(
            &mut v,
            i + task.workloads[state.me].row_start as u64,
            task.is_coset,
            task.is_inv,
            &domain,
            &c_domain,
            &r_domain,
        );

        task.rows[i as usize] = v;

        Promise::ok(())
    }

    fn fft2_prepare(
        &mut self,
        params: plonk_slave::Fft2PrepareParams,
        _: plonk_slave::Fft2PrepareResults,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();
        let id = p.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_tasks.get_mut(&id).unwrap();
        let network = &state.network;
        let me = state.me as u64;

        Promise::from_future(async move {
            let rows = &task.rows;
            let workloads = &task.workloads;

            join_all(
                network
                    .peers
                    .iter()
                    .zip(workloads)
                    .map(|(peer, workload)| async move {
                        let stream = tokio::net::TcpStream::connect(peer).await.unwrap();
                        stream.set_nodelay(true).unwrap();
                        let (reader, writer) =
                            tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                        let mut rpc_system = RpcSystem::new(
                            Box::new(twoparty::VatNetwork::new(
                                reader,
                                writer,
                                rpc_twoparty_capnp::Side::Client,
                                ReaderOptions {
                                    traversal_limit_in_words: Some(usize::MAX),
                                    nesting_limit: 64,
                                },
                            )),
                            None,
                        );
                        let connection = rpc_system
                            .bootstrap::<plonk_peer::Client>(rpc_twoparty_capnp::Side::Server);
                        tokio::task::spawn_local(rpc_system);

                        let mut request = connection.fft_exchange_request();
                        let mut r = request.get();
                        r.set_id(id);
                        r.set_from(me);
                        let v = rows
                            .iter()
                            .flat_map(|i| i[workload.col_start..workload.col_end].to_vec())
                            .collect::<Vec<_>>();
                        let v = serialize(&v);
                        let mut builder = r.init_v(v.len().div_ceil(1 << 28) as u32);
                        for (i, chunk) in v.chunks(1 << 28).enumerate() {
                            builder.set(i as u32, chunk);
                        }

                        request.send().promise.await.unwrap()
                    }),
            )
            .await;

            task.rows = vec![];
            Ok(())
        })
    }

    fn fft2(
        &mut self,
        params: plonk_slave::Fft2Params,
        mut results: plonk_slave::Fft2Results,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();

        let id = p.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_tasks.get_mut(&id).unwrap();

        let (c_domain, r_domain) = if task.is_quot {
            (self.state.quot_c_domain, self.state.quot_r_domain)
        } else {
            (self.state.c_domain, self.state.r_domain)
        };

        let mut builder = results.get().init_v(task.cols.len() as u32);
        for (i, chunk) in task.cols.iter_mut().enumerate() {
            fft2_helper(
                chunk,
                (i + task.workloads[state.me].col_start) as u64,
                task.is_coset,
                task.is_inv,
                &c_domain,
                &r_domain,
            );
            builder.set(i as u32, serialize(&chunk));
        }

        state.fft_tasks.remove(&id);

        Promise::ok(())
    }

    fn round1(
        &mut self,
        params: plonk_slave::Round1Params,
        mut results: plonk_slave::Round1Results,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };

        let mut evals = vec![];
        p.get_w()
            .unwrap()
            .iter()
            .for_each(|n| evals.extend_from_slice(deserialize(n.unwrap())));

        state.domain.ifft_in_place(&mut evals);

        state.wire = DensePolynomial::rand(1, &mut state.rng).mul_by_vanishing_poly(state.domain)
            + DensePolynomial::from_coefficients_vec(evals);

        results
            .get()
            .set_c(serialize(&[commit_polynomial(&state.bases, &state.wire)]));

        Promise::ok(())
    }
}

impl plonk_peer::Server for PlonkImpl {
    fn fft_exchange(
        &mut self,
        params: plonk_peer::FftExchangeParams,
        _: plonk_peer::FftExchangeResults,
    ) -> Promise<(), capnp::Error> {
        let p = params.get().unwrap();

        let mut v = vec![];

        p.get_v()
            .unwrap()
            .iter()
            .for_each(|n| v.extend_from_slice(deserialize(n.unwrap())));

        let from = p.get_from() as usize;
        let id = p.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_tasks.get_mut(&id).unwrap();

        let num_cols = task.workloads[state.me].num_cols();
        for i in 0..v.len() {
            task.cols[i % num_cols][task.workloads[from].row_start + i / num_cols] = v[i];
        }

        Promise::ok(())
    }
}

#[tokio::main(flavor = "current_thread")]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        println!("usage: {} <me>", args[0]);
        return Ok(());
    }

    let me = args[1].parse().unwrap();

    let network: NetworkConfig = serde_json::from_reader(File::open("config/network.json")?)?;

    let local = tokio::task::LocalSet::new();

    let state = Arc::new(State {
        rng: thread_rng(),

        network,
        me,

        bases: vec![],
        domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        c_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        r_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_c_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_r_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),

        fft_tasks: HashMap::new(),

        wire: DensePolynomial { coeffs: vec![] },
    });
    let s = Arc::clone(&state);
    local.spawn_local(async move {
        let listener = tokio::net::TcpListener::bind(s.network.slaves[me])
            .await
            .unwrap();

        loop {
            let (stream, _) = listener.accept().await.unwrap();
            stream.set_nodelay(true).unwrap();
            let (reader, writer) =
                tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
            let network = twoparty::VatNetwork::new(
                reader,
                writer,
                rpc_twoparty_capnp::Side::Server,
                ReaderOptions {
                    traversal_limit_in_words: Some(usize::MAX),
                    nesting_limit: 64,
                },
            );

            tokio::task::spawn_local(RpcSystem::new(
                Box::new(network),
                Some(
                    capnp_rpc::new_client::<plonk_slave::Client, _>(PlonkImpl { state: s.clone() })
                        .client,
                ),
            ));
        }
    });
    let s = state.clone();
    local.spawn_local(async move {
        let listener = tokio::net::TcpListener::bind(s.network.peers[me])
            .await
            .unwrap();

        loop {
            let (stream, _) = listener.accept().await.unwrap();
            stream.set_nodelay(true).unwrap();
            let (reader, writer) =
                tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
            let network = twoparty::VatNetwork::new(
                reader,
                writer,
                rpc_twoparty_capnp::Side::Server,
                ReaderOptions {
                    traversal_limit_in_words: Some(usize::MAX),
                    nesting_limit: 64,
                },
            );

            tokio::task::spawn_local(RpcSystem::new(
                Box::new(network),
                Some(
                    capnp_rpc::new_client::<plonk_peer::Client, _>(PlonkImpl { state: s.clone() })
                        .client,
                ),
            ));
        }
    });

    local.await;
    Ok(())
}
