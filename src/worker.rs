use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::msm::VariableBaseMSM;
use ark_ff::{FftField, Field, PrimeField, ToBytes};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use capnp::capability::Promise;
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};

use hello_world::{
    config::NetworkConfig,
    hello_world_capnp::{fft_peer, plonk},
    utils::{deserialize, serialize, FftWorkload},
};
use tokio::runtime::Handle;

use core::slice;
use futures::{future::join_all, AsyncReadExt};
use std::{
    cell::RefCell,
    collections::HashMap,
    net::{SocketAddr, ToSocketAddrs},
    rc::Rc,
    sync::{Arc, Mutex}, fs::File,
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
    me: usize,
    network: NetworkConfig,

    bases: Vec<G1Affine>,
    domain: Radix2EvaluationDomain<Fr>,
    c_domain: Radix2EvaluationDomain<Fr>,
    r_domain: Radix2EvaluationDomain<Fr>,
    quot_domain: Radix2EvaluationDomain<Fr>,
    quot_c_domain: Radix2EvaluationDomain<Fr>,
    quot_r_domain: Radix2EvaluationDomain<Fr>,

    fft_sessions: HashMap<u64, FftTask>,
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

impl plonk::Server for PlonkImpl {
    fn init(
        &mut self,
        params: plonk::InitParams,
        _: plonk::InitResults,
    ) -> Promise<(), capnp::Error> {
        let domain_size = params.get().unwrap().get_domain_size() as usize;
        let quot_domain_size = params.get().unwrap().get_quot_domain_size() as usize;

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        state.bases = deserialize(params.get().unwrap().get_bases().unwrap()).to_vec();
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
        params: plonk::VarMsmParams,
        mut results: plonk::VarMsmResults,
    ) -> Promise<(), capnp::Error> {
        let scalars = deserialize(params.get().unwrap().get_scalars().unwrap());

        results
            .get()
            .set_result(serialize(&[VariableBaseMSM::multi_scalar_mul(
                &self.state.bases,
                scalars,
            )]));

        Promise::ok(())
    }

    // fn fft1(
    //     &mut self,
    //     params: plonk::Fft1Params,
    //     mut results: plonk::Fft1Results,
    // ) -> Promise<(), capnp::Error> {
    //     let v = &mut deserialize::<Fr>(params.get().unwrap().get_v().unwrap()).to_vec();
    //     let i = params.get().unwrap().get_i();
    //     let is_quot = params.get().unwrap().get_is_quot();
    //     let is_inv = params.get().unwrap().get_is_inv();
    //     let is_coset = params.get().unwrap().get_is_coset();

    //     let (domain, c_domain, r_domain) = if is_quot {
    //         (
    //             self.state.quot_domain,
    //             self.state.quot_c_domain,
    //             self.state.quot_r_domain,
    //         )
    //     } else {
    //         (self.state.domain, self.state.c_domain, self.state.r_domain)
    //     };

    //     if is_coset && !is_inv {
    //         let g = Fr::multiplicative_generator();
    //         v.iter_mut()
    //             .enumerate()
    //             .for_each(|(j, u)| *u *= g.pow([i + j as u64 * r_domain.size]));
    //     }
    //     if is_inv {
    //         c_domain.ifft_in_place(v);
    //     } else {
    //         c_domain.fft_in_place(v);
    //     }
    //     let omega_shift = if is_inv {
    //         domain.group_gen_inv
    //     } else {
    //         domain.group_gen
    //     };
    //     v.iter_mut()
    //         .enumerate()
    //         .for_each(|(j, u)| *u *= omega_shift.pow([i * j as u64]));

    //     results.get().set_v(serialize(v));

    //     Promise::ok(())
    // }

    // fn fft2(
    //     &mut self,
    //     params: plonk::Fft2Params,
    //     mut results: plonk::Fft2Results,
    // ) -> Promise<(), capnp::Error> {
    //     let v = &mut deserialize::<Fr>(params.get().unwrap().get_v().unwrap()).to_vec();
    //     let i = params.get().unwrap().get_i();
    //     let is_quot = params.get().unwrap().get_is_quot();
    //     let is_inv = params.get().unwrap().get_is_inv();
    //     let is_coset = params.get().unwrap().get_is_coset();

    //     let (c_domain, r_domain) = if is_quot {
    //         (self.state.quot_c_domain, self.state.quot_r_domain)
    //     } else {
    //         (self.state.c_domain, self.state.r_domain)
    //     };
    //     if is_inv {
    //         r_domain.ifft_in_place(v);
    //     } else {
    //         r_domain.fft_in_place(v);
    //     }
    //     if is_coset && is_inv {
    //         let g = Fr::multiplicative_generator().inverse().unwrap();
    //         v.iter_mut()
    //             .enumerate()
    //             .for_each(|(j, u)| *u *= g.pow([i + j as u64 * c_domain.size]));
    //     }

    //     results.get().set_v(serialize(v));

    //     Promise::ok(())
    // }

    fn fft_init(
        &mut self,
        params: plonk::FftInitParams,
        _: plonk::FftInitResults,
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
        state.fft_sessions.insert(
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
        params: plonk::Fft1Params,
        _: plonk::Fft1Results,
    ) -> Promise<(), capnp::Error> {
        let params = params.get().unwrap();

        let mut v = deserialize::<Fr>(params.get_v().unwrap()).to_vec();
        let i = params.get_i();
        let id = params.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_sessions.get_mut(&id).unwrap();

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
        params: plonk::Fft2PrepareParams,
        _: plonk::Fft2PrepareResults,
    ) -> Promise<(), capnp::Error> {
        let params = params.get().unwrap();
        let id = params.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_sessions.get_mut(&id).unwrap();
        let network = &state.network;
        let me = state.me as u64;

        Promise::from_future(async move {
            let rows = &task.rows;
            let workloads = &task.workloads;

            join_all((0..network.peers.len()).map(|peer| async move {
                let stream = tokio::net::TcpStream::connect(network.peers[peer])
                    .await
                    .unwrap();
                stream.set_nodelay(true).unwrap();
                let (reader, writer) =
                    tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
                let mut rpc_system = RpcSystem::new(
                    Box::new(twoparty::VatNetwork::new(
                        reader,
                        writer,
                        rpc_twoparty_capnp::Side::Client,
                        Default::default(),
                    )),
                    None,
                );
                let connection: fft_peer::Client =
                    rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
                tokio::task::spawn_local(rpc_system);

                let mut request = connection.exchange_request();
                request.get().set_id(id);
                request.get().set_from(me);
                request.get().set_v(serialize(
                    &rows
                        .iter()
                        .flat_map(|i| {
                            i[workloads[peer].col_start..workloads[peer].col_end].to_vec()
                        })
                        .collect::<Vec<_>>(),
                ));
                request.send().promise.await.unwrap()
            }))
            .await;

            task.rows = vec![];
            Ok(())
        })
    }

    fn fft2(
        &mut self,
        params: plonk::Fft2Params,
        mut results: plonk::Fft2Results,
    ) -> Promise<(), capnp::Error> {
        let params = params.get().unwrap();

        let id = params.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_sessions.get_mut(&id).unwrap();

        let (c_domain, r_domain) = if task.is_quot {
            (self.state.quot_c_domain, self.state.quot_r_domain)
        } else {
            (self.state.c_domain, self.state.r_domain)
        };

        for i in 0..task.cols.len() {
            fft2_helper(
                &mut task.cols[i],
                (i + task.workloads[state.me].col_start) as u64,
                task.is_coset,
                task.is_inv,
                &c_domain,
                &r_domain,
            );
        }

        results.get().set_v(serialize(
            &task.cols.clone().into_iter().flatten().collect::<Vec<_>>(),
        ));

        state.fft_sessions.remove(&id);

        Promise::ok(())
    }
}

impl fft_peer::Server for PlonkImpl {
    fn exchange(
        &mut self,
        params: fft_peer::ExchangeParams,
        _: fft_peer::ExchangeResults,
    ) -> Promise<(), capnp::Error> {
        let params = params.get().unwrap();

        let v = deserialize::<Fr>(params.get_v().unwrap()).to_vec();
        let from = params.get_from() as usize;
        let id = params.get_id();

        let state = unsafe { &mut *(Arc::as_ptr(&self.state) as *mut State) };
        let task = state.fft_sessions.get_mut(&id).unwrap();

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
        network,
        me,

        bases: vec![],
        domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        c_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        r_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_c_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),
        quot_r_domain: Radix2EvaluationDomain::<Fr>::new(1).unwrap(),

        fft_sessions: HashMap::new(),
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
                Default::default(),
            );

            tokio::task::spawn_local(RpcSystem::new(
                Box::new(network),
                Some(
                    capnp_rpc::new_client::<plonk::Client, _>(PlonkImpl { state: s.clone() })
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
                Default::default(),
            );

            tokio::task::spawn_local(RpcSystem::new(
                Box::new(network),
                Some(
                    capnp_rpc::new_client::<fft_peer::Client, _>(PlonkImpl { state: s.clone() })
                        .client,
                ),
            ));
        }
    });

    local.await;
    Ok(())
}
