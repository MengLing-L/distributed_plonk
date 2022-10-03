#![feature(int_roundings)]

use std::{
    cmp::{max, min},
    error::Error,
    fs::File,
    net::ToSocketAddrs,
};

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::msm::VariableBaseMSM;
use ark_ff::{FromBytes, PrimeField, ToBytes, UniformRand, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use capnp::{message::ReaderOptions, private::layout::PrimitiveElement};
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::{future::join_all, AsyncReadExt};
use hello_world::{
    config::NetworkConfig,
    hello_world_capnp::{plonk_peer, plonk_slave},
    transpose::ip_transpose,
    utils::{deserialize, serialize, FftWorkload, MsmWorkload},
};
use jf_primitives::circuit::merkle_tree::{AccElemVars, AccMemberWitnessVar, MerkleTreeGadget};
use jf_primitives::merkle_tree::{AccMemberWitness, MerkleTree};
use rand::thread_rng;

pub async fn connect<A: tokio::net::ToSocketAddrs>(
    addr: A,
) -> Result<plonk_slave::Client, Box<dyn Error>> {
    let stream = tokio::net::TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
    let rpc_network = Box::new(twoparty::VatNetwork::new(
        reader,
        writer,
        rpc_twoparty_capnp::Side::Client,
        ReaderOptions {
            traversal_limit_in_words: Some(usize::MAX),
            nesting_limit: 64,
        },
    ));
    let mut rpc_system = RpcSystem::new(rpc_network, None);
    let client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
    tokio::task::spawn_local(rpc_system);
    Ok(client)
}

pub async fn init(
    connection: &plonk_slave::Client,
    bases: &[G1Affine],
    domain_size: usize,
    quot_domain_size: usize,
) -> Result<(), Box<dyn Error>> {
    let mut request = connection.init_request();
    let mut r = request.get();
    r.set_domain_size(domain_size as u64);
    r.set_quot_domain_size(quot_domain_size as u64);
    let bases = serialize(bases);
    let mut builder = r.init_bases(bases.len().div_ceil(1 << 28) as u32);
    for (i, chunk) in bases.chunks(1 << 28).enumerate() {
        builder.set(i as u32, chunk);
    }

    request.send().promise.await?;
    Ok(())
}

pub async fn msm(
    connection: &plonk_slave::Client,
    workload: &MsmWorkload,
    scalars: &[<Fr as PrimeField>::BigInt],
) -> Result<G1Projective, Box<dyn Error>> {
    let mut request = connection.var_msm_request();

    let mut w = request.get().init_workload();
    w.set_start(workload.start as u64);
    w.set_end(workload.end as u64);
    let scalars = serialize(scalars);
    let mut builder = request
        .get()
        .init_scalars(scalars.len().div_ceil(1 << 28) as u32);
    for (i, chunk) in scalars.chunks(1 << 28).enumerate() {
        builder.set(i as u32, chunk);
    }

    let reply = request.send().promise.await?;
    let r = reply.get()?.get_result()?;

    Ok(deserialize(r)[0])
}

pub async fn fft_init(
    connection: &plonk_slave::Client,
    id: u64,
    workloads: &[FftWorkload],
    is_quot: bool,
    is_inv: bool,
    is_coset: bool,
) -> Result<(), Box<dyn Error>> {
    let mut request = connection.fft_init_request();
    let mut r = request.get();
    r.set_id(id);
    r.set_is_coset(is_coset);
    r.set_is_inv(is_inv);
    r.set_is_quot(is_quot);
    let mut w = r.init_workloads(workloads.len() as u32);
    for i in 0..workloads.len() {
        w.reborrow()
            .get(i as u32)
            .set_row_start(workloads[i].row_start as u64);
        w.reborrow()
            .get(i as u32)
            .set_row_end(workloads[i].row_end as u64);
        w.reborrow()
            .get(i as u32)
            .set_col_start(workloads[i].col_start as u64);
        w.reborrow()
            .get(i as u32)
            .set_col_end(workloads[i].col_end as u64);
    }

    request.send().promise.await?;

    Ok(())
}

pub async fn fft1(
    connection: &plonk_slave::Client,
    id: u64,
    i: usize,
    v: &[Fr],
) -> Result<(), Box<dyn Error>> {
    let mut request = connection.fft1_request();
    let mut r = request.get();

    r.set_i(i as u64);
    r.set_id(id);

    let v = serialize(v);
    let mut builder = r.init_v(v.len().div_ceil(1 << 28) as u32);
    for (i, chunk) in v.chunks(1 << 28).enumerate() {
        builder.set(i as u32, chunk);
    }

    request.send().promise.await?;

    Ok(())
}

pub async fn fft2_prepare(connection: &plonk_slave::Client, id: u64) -> Result<(), Box<dyn Error>> {
    let mut request = connection.fft2_prepare_request();
    request.get().set_id(id);

    request.send().promise.await?;

    Ok(())
}

pub async fn fft2(connection: &plonk_slave::Client, id: u64) -> Result<Vec<Fr>, Box<dyn Error>> {
    let mut request = connection.fft2_request();
    request.get().set_id(id);
    let reply = request.send().promise.await?;

    let mut r = vec![];

    reply
        .get()?
        .get_v()?
        .iter()
        .for_each(|n| r.extend_from_slice(deserialize(n.unwrap())));

    Ok(r)
}

#[test]
pub fn test_msm() -> Result<(), Box<dyn Error>> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let network: NetworkConfig = serde_json::from_reader(File::open("config/network.json")?)?;
    let num_slaves = network.slaves.len();

    let rng = &mut thread_rng();

    let l = 1 << 20;

    let mut bases = (0..min(l, 1 << 11))
        .map(|_| G1Projective::rand(rng).into())
        .collect::<Vec<G1Affine>>();

    while bases.len() < l {
        bases.append(&mut bases.clone());
    }

    let exps = (0..l)
        .map(|_| Fr::rand(rng).into_repr())
        .collect::<Vec<_>>();

    tokio::task::LocalSet::new().block_on(&rt, async move {
        let slaves = join_all(
            network
                .slaves
                .iter()
                .map(|addr| async move { connect(addr).await.unwrap() }),
        )
        .await;

        let bases = &bases;

        join_all(slaves.iter().map(|slave| async move {
            init(slave, bases, 0, 0).await.unwrap();
        }))
        .await;

        let r = join_all(
            exps.chunks(exps.len() / num_slaves)
                .enumerate()
                .map(|(i, scalars)| {
                    (
                        MsmWorkload {
                            start: i * exps.len() / num_slaves,
                            end: (i + 1) * exps.len() / num_slaves,
                        },
                        scalars,
                    )
                })
                .zip(&slaves)
                .map(|((workload, scalars), connection)| async move {
                    msm(&connection, &workload, scalars).await.unwrap()
                }),
        )
        .await
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap();

        assert_eq!(r, VariableBaseMSM::multi_scalar_mul(bases, &exps));

        Ok(())
    })
}

#[tokio::test]
pub async fn test_fft() -> Result<(), Box<dyn Error>> {
    let network: NetworkConfig = serde_json::from_reader(File::open("config/network.json")?)?;
    let num_slaves = network.slaves.len();

    let rng = &mut thread_rng();

    let domain = Radix2EvaluationDomain::<Fr>::new(1 << 11).unwrap();
    let quot_domain = Radix2EvaluationDomain::<Fr>::new(1 << 13).unwrap();

    tokio::task::LocalSet::new()
        .run_until(async move {
            let connections = join_all(
                network
                    .slaves
                    .iter()
                    .map(|addr| async move { connect(addr).await.unwrap() }),
            )
            .await;

            join_all(connections.iter().map(|connection| async move {
                init(connection, &[], domain.size(), quot_domain.size())
                    .await
                    .unwrap();
            }))
            .await;

            for is_quot in [false, true] {
                let domain = if is_quot { quot_domain } else { domain };

                let r = 1 << (domain.log_size_of_group >> 1);
                let c = domain.size() / r;

                let workloads = &(0..num_slaves)
                    .map(|i| FftWorkload {
                        row_start: i * r / num_slaves,
                        row_end: (i + 1) * r / num_slaves,
                        col_start: i * c / num_slaves,
                        col_end: (i + 1) * c / num_slaves,
                    })
                    .collect::<Vec<_>>();

                let coeffs = (0..domain.size())
                    .map(|_| Fr::rand(rng))
                    .collect::<Vec<_>>();

                for is_inv in [false, true] {
                    for is_coset in [false, true] {
                        let id: u64 = rng.gen();

                        join_all(connections.iter().map(|connection| async move {
                            fft_init(connection, id, workloads, is_quot, is_inv, is_coset)
                                .await
                                .unwrap()
                        }))
                        .await;

                        let mut t = coeffs.clone();
                        let mut w = vec![Fr::zero(); r];
                        ip_transpose(&mut t, &mut w, c, r);

                        join_all(
                            connections
                                .iter()
                                .zip(t.chunks(domain.size() / num_slaves))
                                .map(|(connection, tt)| async move {
                                    join_all(tt.chunks(c).enumerate().map(|(j, v)| async move {
                                        fft1(connection, id, j, v).await.unwrap()
                                    }))
                                    .await
                                }),
                        )
                        .await;
                        join_all(connections.iter().map(|connection| async move {
                            fft2_prepare(connection, id).await.unwrap()
                        }))
                        .await;
                        let mut u = vec![];
                        let t =
                            join_all(connections.iter().map(|connection| async move {
                                fft2(connection, id).await.unwrap()
                            }))
                            .await;
                        for i in 0..num_slaves {
                            u.extend_from_slice(&t[i]);
                        }
                        ip_transpose(&mut u, &mut w, c, r);

                        assert_eq!(
                            u,
                            match (is_inv, is_coset) {
                                (true, true) => domain.coset_ifft(&coeffs),
                                (true, false) => domain.ifft(&coeffs),
                                (false, true) => domain.coset_fft(&coeffs),
                                (false, false) => domain.fft(&coeffs),
                            }
                        );
                    }
                }
            }

            Ok(())
        })
        .await
}

fn main() {}

use std::ops::Add;

use ark_bls12_381::Bls12_381;
use ark_ff::{FftField, Field, One};
use ark_poly::{univariate::DensePolynomial, Polynomial, UVPolynomial};
use ark_std::{
    format,
    ops::Mul,
    println,
    rand::{CryptoRng, RngCore},
    time::Instant,
    vec,
    vec::Vec,
};
use jf_plonk::{
    circuit::{gates::Gate, Arithmetization, GateId, Variable, WireId},
    constants::GATE_WIDTH,
    errors::{CircuitError, SnarkError},
    prelude::{
        Circuit, PlonkCircuit, PlonkError, Proof, ProofEvaluations, ProvingKey, VerifyingKey,
    },
    proof_system::{PlonkKzgSnark, Snark}, transcript::StandardTranscript,
};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use ark_ec::{
    short_weierstrass_jacobian::GroupAffine, PairingEngine, SWModelParameters as SWParam,
};
use ark_poly_commit::kzg10::Commitment;
use jf_utils::to_bytes;
use merlin::Transcript;

/// A wrapper of `merlin::Transcript`.
struct FakeStandardTranscript(Transcript);

impl FakeStandardTranscript {
    /// create a new plonk transcript
    pub fn new(label: &'static [u8]) -> Self {
        Self(Transcript::new(label))
    }

    /// Append the verification key and the public input to the transcript.
    pub fn append_vk_and_pub_input<F, E, P>(
        &mut self,
        vk: &VerifyingKey<E>,
        pub_input: &[E::Fr],
    ) -> Result<(), PlonkError>
    where
        E: PairingEngine<Fq = F, G1Affine = GroupAffine<P>>,
        P: SWParam<BaseField = F> + Clone,
    {
        self.0.append_message(
            b"field size in bits",
            E::Fr::size_in_bits().to_le_bytes().as_ref(),
        );
        self.0
            .append_message(b"domain size", vk.domain_size.to_le_bytes().as_ref());
        self.0
            .append_message(b"input size", vk.num_inputs.to_le_bytes().as_ref());

        for ki in vk.k.iter() {
            self.0
                .append_message(b"wire subsets separators", &to_bytes!(ki)?);
        }
        for selector_com in vk.selector_comms.iter() {
            self.0
                .append_message(b"selector commitments", &to_bytes!(selector_com)?);
        }

        for sigma_comms in vk.sigma_comms.iter() {
            self.0
                .append_message(b"sigma commitments", &to_bytes!(sigma_comms)?);
        }

        for input in pub_input.iter() {
            self.0.append_message(b"public input", &to_bytes!(input)?);
        }

        Ok(())
    }

    /// Append a slice of commitments to the transcript.
    pub fn append_commitments<F, E, P>(
        &mut self,
        label: &'static [u8],
        comms: &[Commitment<E>],
    ) -> Result<(), PlonkError>
    where
        E: PairingEngine<Fq = F, G1Affine = GroupAffine<P>>,
        P: SWParam<BaseField = F> + Clone,
    {
        for comm in comms.iter() {
            self.0.append_message(label, &to_bytes!(comm)?);
        }
        Ok(())
    }

    /// Append a single commitment to the transcript.
    pub fn append_commitment<F, E, P>(
        &mut self,
        label: &'static [u8],
        comm: &Commitment<E>,
    ) -> Result<(), PlonkError>
    where
        E: PairingEngine<Fq = F, G1Affine = GroupAffine<P>>,
        P: SWParam<BaseField = F> + Clone,
    {
        self.0.append_message(label, &to_bytes!(comm)?);
        Ok(())
    }

    /// Append a proof evaluation to the transcript.
    pub fn append_proof_evaluations<E: PairingEngine>(
        &mut self,
        wires_evals: &[E::Fr],
        wire_sigma_evals: &[E::Fr],
        perm_next_eval: &E::Fr,
    ) -> Result<(), PlonkError> {
        for w_eval in wires_evals {
            self.0.append_message(b"wire_evals", &to_bytes!(w_eval)?);
        }
        for sigma_eval in wire_sigma_evals {
            self.0
                .append_message(b"wire_sigma_evals", &to_bytes!(sigma_eval)?);
        }
        self.0
            .append_message(b"perm_next_eval", &to_bytes!(perm_next_eval)?);
        Ok(())
    }

    // generate the challenge for the current transcript
    // and append it to the transcript
    pub fn get_and_append_challenge<E>(&mut self, label: &'static [u8]) -> Result<E::Fr, PlonkError>
    where
        E: PairingEngine,
    {
        let mut buf = [0u8; 64];
        self.0.challenge_bytes(label, &mut buf);
        let challenge = E::Fr::from_le_bytes_mod_order(&buf);
        self.0.append_message(label, &to_bytes!(&challenge)?);
        Ok(challenge)
    }
}

/// A specific Plonk circuit instantiation.
#[derive(Debug, Clone)]
pub struct FakePlonkCircuit<F>
where
    F: FftField,
{
    num_vars: usize,
    gates: Vec<Box<dyn Gate<F>>>,
    wire_variables: [Vec<Variable>; GATE_WIDTH + 2],
    pub_input_gate_ids: Vec<GateId>,
    witness: Vec<F>,
    wire_permutation: Vec<(WireId, GateId)>,
    extended_id_permutation: Vec<F>,
    num_wire_types: usize,
    eval_domain: Radix2EvaluationDomain<F>,
}

pub struct Prover {}

struct Context {
    slaves: Vec<plonk_slave::Client>,
}

impl Prover {
    async fn prove_async<C, R>(
        ctx: &Context,
        prng: &mut R,
        circuit: &C,
        prove_key: &ProvingKey<'_, Bls12_381>,
    ) -> Result<Proof<Bls12_381>, PlonkError>
    where
        C: Arithmetization<Fr>,
        R: CryptoRng + RngCore,
    {
        // Dirty hack: extract private fields from `circuit`
        let circuit = unsafe { &*(circuit as *const _ as *const FakePlonkCircuit<Fr>) };

        let domain = circuit.eval_domain;
        let n = domain.size();
        let num_wire_types = circuit.num_wire_types;

        let mut ck = prove_key.commit_key.powers_of_g.to_vec();
        ck.resize(((ck.len() + 31) >> 5) << 5, G1Affine::zero());

        if n == 1 {
            return Err(CircuitError::UnfinalizedCircuit.into());
        }
        if n < circuit.gates.len() {
            return Err(SnarkError::ParameterError(format!(
                "Domain size {} should be bigger than number of constraint {}",
                n,
                circuit.gates.len()
            ))
            .into());
        }
        if prove_key.domain_size() != n {
            return Err(SnarkError::ParameterError(format!(
                "proving key domain size {} != expected domain size {}",
                prove_key.domain_size(),
                n
            ))
            .into());
        }
        if circuit.pub_input_gate_ids.len() != prove_key.vk.num_inputs {
            return Err(SnarkError::ParameterError(format!(
                "circuit.num_inputs {} != prove_key.num_inputs {}",
                circuit.pub_input_gate_ids.len(),
                prove_key.vk.num_inputs
            ))
            .into());
        }

        let bases = &ck;
        join_all(ctx.slaves.iter().map(|slave| async move {
            init(slave, bases, domain.size(), domain.size())
                .await
                .unwrap();
        }))
        .await;

        // Initialize transcript
        let mut transcript = FakeStandardTranscript::new(b"PlonkProof");
        let mut pub_input = DensePolynomial::from_coefficients_vec(
            circuit
                .pub_input_gate_ids
                .iter()
                .map(|&gate_id| {
                    circuit.witness[circuit.wire_variables[num_wire_types - 1][gate_id]]
                })
                .collect(),
        );

        transcript.append_vk_and_pub_input(&prove_key.vk, &pub_input)?;

        domain.ifft_in_place(&mut pub_input.coeffs);
        // Self::ifft(&kernel, &domain, &mut pub_input);

        let comms = join_all(
            circuit
                .wire_variables
                .iter()
                .take(num_wire_types)
                .zip(&ctx.slaves)
                .map(|(wire_vars, slave)| async move {
                    let mut request = slave.round1_request();

                    let wire = wire_vars
                        .iter()
                        .map(|&var| circuit.witness[var])
                        .collect::<Vec<_>>();
                    let wire = serialize(&wire);
                    let mut builder = request.get().init_w(wire.len().div_ceil(1 << 28) as u32);
                    for (i, chunk) in wire.chunks(1 << 28).enumerate() {
                        builder.set(i as u32, chunk);
                    }

                    let reply = request.send().promise.await.unwrap();
                    let r = reply.get().unwrap().get_c().unwrap();

                    Commitment::<Bls12_381>(deserialize::<G1Projective>(r)[0].into())
                }),
        )
        .await;

        // Round 1
        let now = Instant::now();
        let wire_polys = circuit
            .wire_variables
            .iter()
            .take(num_wire_types)
            .map(|wire_vars| {
                let mut coeffs = wire_vars.iter().map(|&var| circuit.witness[var]).collect();
                domain.ifft_in_place(&mut coeffs);
                // Self::ifft(&kernel, &domain, &mut coeffs);
                DensePolynomial::rand(1, prng).mul_by_vanishing_poly(domain)
                    + DensePolynomial::from_coefficients_vec(coeffs)
            })
            .collect::<Vec<_>>();
        let wires_poly_comms = wire_polys
            .iter()
            .map(|poly| {
                Self::commit_polynomial(/* context, */ &ck, poly)
            })
            .collect::<Vec<_>>();
        transcript.append_commitments(b"witness_poly_comms", &wires_poly_comms)?;
        println!("Elapsed: {:.2?}", now.elapsed());

        // Round 2
        let now = Instant::now();
        let beta = transcript.get_and_append_challenge::<Bls12_381>(b"beta")?;
        let gamma = transcript.get_and_append_challenge::<Bls12_381>(b"gamma")?;
        let permutation_poly = {
            let mut product_vec = vec![Fr::one()];
            for j in 0..(n - 1) {
                // Nominator
                let mut a = Fr::one();
                // Denominator
                let mut b = Fr::one();
                for i in 0..num_wire_types {
                    let wire_value = circuit.witness[circuit.wire_variables[i][j]];
                    let tmp = wire_value + gamma;
                    a *= tmp + beta * circuit.extended_id_permutation[i * n + j];
                    let (perm_i, perm_j) = circuit.wire_permutation[i * n + j];
                    b *= tmp + beta * circuit.extended_id_permutation[perm_i * n + perm_j];
                }
                product_vec.push(product_vec[j] * a / b);
            }
            domain.ifft_in_place(&mut product_vec);
            // Self::ifft(&kernel, &domain, &mut product_vec);
            DensePolynomial::rand(2, prng).mul_by_vanishing_poly(domain)
                + DensePolynomial::from_coefficients_vec(product_vec)
        };
        let prod_perm_poly_comm = Self::commit_polynomial(
            // context,
            &ck,
            &permutation_poly,
        );
        transcript.append_commitment(b"perm_poly_comms", &prod_perm_poly_comm)?;
        println!("Elapsed: {:.2?}", now.elapsed());

        // Round 3
        let now = Instant::now();
        let alpha = transcript.get_and_append_challenge::<Bls12_381>(b"alpha")?;
        let alpha_square_div_n = alpha.square() / Fr::from(n as u64);
        let quotient_poly = {
            let tmp_domain = Radix2EvaluationDomain::<Fr>::new((n + 2) * 5).unwrap();

            let ab = wire_polys[0].mul(&wire_polys[1]);
            let cd = wire_polys[2].mul(&wire_polys[3]);

            let mut f = &pub_input
                + &prove_key.selectors[11]
                + prove_key.selectors[0].mul(&wire_polys[0])
                + prove_key.selectors[1].mul(&wire_polys[1])
                + prove_key.selectors[2].mul(&wire_polys[2])
                + prove_key.selectors[3].mul(&wire_polys[3])
                + prove_key.selectors[4].mul(&ab)
                + prove_key.selectors[5].mul(&cd)
                + prove_key.selectors[6].mul(&{
                    let mut evals = tmp_domain.fft(&wire_polys[0]);
                    evals.iter_mut().for_each(|x| *x *= x.square().square());
                    tmp_domain.ifft_in_place(&mut evals);
                    DensePolynomial::from_coefficients_vec(evals)
                })
                + prove_key.selectors[7].mul(&{
                    let mut evals = tmp_domain.fft(&wire_polys[1]);
                    evals.iter_mut().for_each(|x| *x *= x.square().square());
                    tmp_domain.ifft_in_place(&mut evals);
                    DensePolynomial::from_coefficients_vec(evals)
                })
                + prove_key.selectors[8].mul(&{
                    let mut evals = tmp_domain.fft(&wire_polys[2]);
                    evals.iter_mut().for_each(|x| *x *= x.square().square());
                    tmp_domain.ifft_in_place(&mut evals);
                    DensePolynomial::from_coefficients_vec(evals)
                })
                + prove_key.selectors[9].mul(&{
                    let mut evals = tmp_domain.fft(&wire_polys[3]);
                    evals.iter_mut().for_each(|x| *x *= x.square().square());
                    tmp_domain.ifft_in_place(&mut evals);
                    DensePolynomial::from_coefficients_vec(evals)
                })
                + -prove_key.selectors[10].mul(&wire_polys[4])
                + prove_key.selectors[12]
                    .mul(&ab)
                    .mul(&cd)
                    .mul(&wire_polys[4]);

            let mut g = permutation_poly.mul(alpha);
            for i in 0..num_wire_types {
                g = g.mul(
                    &(&wire_polys[i]
                        + &DensePolynomial {
                            coeffs: vec![gamma, beta * prove_key.vk.k[i]],
                        }),
                );
            }
            f = f + g;

            let mut h = permutation_poly.mul(-alpha);
            {
                let mut t = Fr::one();
                for i in 0..h.len() {
                    h[i] *= t;
                    t *= domain.group_gen;
                }
            }
            for i in 0..num_wire_types {
                h = h.mul(
                    &(&wire_polys[i]
                        + &prove_key.sigmas[i].mul(beta)
                        + DensePolynomial {
                            coeffs: vec![gamma],
                        }),
                );
            }
            f = f + h;

            ({
                let mut remainder = f;
                let mut quotient = vec![Fr::zero(); remainder.degree()];

                while !remainder.is_zero() && remainder.degree() >= n {
                    let cur_q_coeff = *remainder.coeffs.last().unwrap();
                    let cur_q_degree = remainder.degree() - n;
                    quotient[cur_q_degree] = cur_q_coeff;

                    remainder[cur_q_degree] += &cur_q_coeff;
                    remainder[cur_q_degree + n] -= &cur_q_coeff;
                    while let Some(true) = remainder.coeffs.last().map(|c| c.is_zero()) {
                        remainder.coeffs.pop();
                    }
                }
                DensePolynomial::from_coefficients_vec(quotient)
            } + {
                let mut r = permutation_poly.mul(alpha_square_div_n);
                r[0] -= alpha_square_div_n;
                let mut t = r.coeffs.pop().unwrap();
                for i in (0..r.len()).rev() {
                    (r[i], t) = (t, r[i] + t);
                }
                r
            })
        };
        let split_quot_polys = {
            let expected_degree = num_wire_types * (n + 1) + 2;
            if quotient_poly.degree() != expected_degree {
                return Err(SnarkError::WrongQuotientPolyDegree(
                    quotient_poly.degree(),
                    expected_degree,
                )
                .into());
            }
            quotient_poly
                .coeffs
                .chunks(n + 2)
                .map(DensePolynomial::from_coefficients_slice)
                .collect::<Vec<_>>()
        };
        let split_quot_poly_comms = split_quot_polys
            .iter()
            .map(|poly| {
                Self::commit_polynomial(/* context, */ &ck, poly)
            })
            .collect::<Vec<_>>();
        transcript.append_commitments(b"quot_poly_comms", &split_quot_poly_comms)?;
        println!("Elapsed: {:.2?}", now.elapsed());

        // Round 4
        let now = Instant::now();
        let zeta = transcript.get_and_append_challenge::<Bls12_381>(b"zeta")?;
        let wires_evals = wire_polys
            .par_iter()
            .map(|poly| poly.evaluate(&zeta))
            .collect::<Vec<_>>();
        let wire_sigma_evals = prove_key
            .sigmas
            .par_iter()
            .take(num_wire_types - 1)
            .map(|poly| poly.evaluate(&zeta))
            .collect::<Vec<_>>();
        let perm_next_eval = permutation_poly.evaluate(&(zeta * domain.group_gen));
        transcript.append_proof_evaluations::<Bls12_381>(
            &wires_evals,
            &wire_sigma_evals,
            &perm_next_eval,
        )?;
        println!("Elapsed: {:.2?}", now.elapsed());

        // Round 5
        let now = Instant::now();
        let vanish_eval = zeta.pow(&[n as u64]) - Fr::one();
        let lin_poly = {
            // The selectors order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
            // TODO: (binyi) get the order from a function.
            let q_lc = &prove_key.selectors[..GATE_WIDTH];
            let q_mul = &prove_key.selectors[GATE_WIDTH..GATE_WIDTH + 2];
            let q_hash = &prove_key.selectors[GATE_WIDTH + 2..2 * GATE_WIDTH + 2];
            let q_o = &prove_key.selectors[2 * GATE_WIDTH + 2];
            let q_c = &prove_key.selectors[2 * GATE_WIDTH + 3];
            let q_ecc = &prove_key.selectors[2 * GATE_WIDTH + 4];

            // TODO(binyi): add polynomials in parallel.
            // Note we don't need to compute the constant term of the polynomial.
            let a = wires_evals[0];
            let b = wires_evals[1];
            let c = wires_evals[2];
            let d = wires_evals[3];
            let e = wires_evals[4];
            let ab = a * b;
            let cd = c * d;
            q_lc[0].mul(a)
                + q_lc[1].mul(b)
                + q_lc[2].mul(c)
                + q_lc[3].mul(d)
                + q_mul[0].mul(ab)
                + q_mul[1].mul(cd)
                + q_hash[0].mul(a.square().square() * a)
                + q_hash[1].mul(b.square().square() * b)
                + q_hash[2].mul(c.square().square() * c)
                + q_hash[3].mul(d.square().square() * d)
                + q_ecc.mul(ab * cd * e)
                + q_o.mul(-e)
                + q_c.clone()
        } + {
            let lagrange_1_eval = vanish_eval / (Fr::from(n as u32) * (zeta - Fr::one()));

            // Compute the coefficient of z(X)
            let coeff = wires_evals
                .iter()
                .zip(&prove_key.vk.k)
                .fold(alpha, |acc, (&wire_eval, &k)| {
                    acc * (wire_eval + beta * k * zeta + gamma)
                })
                + alpha.square() * lagrange_1_eval;
            permutation_poly.mul(coeff)
        } + {
            // Compute the coefficient of the last sigma wire permutation polynomial
            let coeff = -wires_evals
                .iter()
                .take(num_wire_types - 1)
                .zip(&wire_sigma_evals)
                .fold(
                    alpha * beta * perm_next_eval,
                    |acc, (&wire_eval, &sigma_eval)| acc * (wire_eval + beta * sigma_eval + gamma),
                );
            prove_key.sigmas[num_wire_types - 1].mul(coeff)
        } + {
            let zeta_to_n_plus_2 = (vanish_eval + Fr::one()) * zeta.square();
            let mut r_quot = split_quot_polys[0].clone();
            let mut coeff = Fr::one();
            for poly in &split_quot_polys[1..] {
                coeff *= zeta_to_n_plus_2;
                r_quot = r_quot + poly.mul(coeff);
            }
            r_quot.mul(-vanish_eval)
        };
        let v = transcript.get_and_append_challenge::<Bls12_381>(b"v")?;

        let opening_proof = {
            // List the polynomials to be opened at point `zeta`.
            let mut polys_ref = vec![&lin_poly];
            for poly in wire_polys.iter() {
                polys_ref.push(poly);
            }
            // Note we do not add the last wire sigma polynomial.
            for poly in prove_key.sigmas.iter().take(prove_key.sigmas.len() - 1) {
                polys_ref.push(poly);
            }
            let (batch_poly, _) = polys_ref.iter().fold(
                (DensePolynomial::zero(), Fr::one()),
                |(acc, coeff), &poly| (acc + poly.mul(coeff), coeff * v),
            );

            Self::commit_polynomial(
                // context,
                &ck,
                &{
                    let mut opening_poly = batch_poly;
                    let mut t = opening_poly.coeffs.pop().unwrap();
                    for i in (0..opening_poly.len()).rev() {
                        (opening_poly[i], t) = (t, opening_poly[i] + t * zeta);
                    }
                    opening_poly
                },
            )
        };

        let shifted_opening_proof = {
            Self::commit_polynomial(
                // context,
                &ck,
                &{
                    let mut opening_poly = permutation_poly;
                    let mut t = opening_poly.coeffs.pop().unwrap();
                    for i in (0..opening_poly.len()).rev() {
                        (opening_poly[i], t) = (t, opening_poly[i] + t * domain.group_gen * zeta);
                    }
                    opening_poly
                },
            )
        };
        println!("Elapsed: {:.2?}", now.elapsed());

        // unsafe {
        //     mult_pippenger_free(context);
        // }

        Ok(Proof {
            wires_poly_comms,
            prod_perm_poly_comm,
            split_quot_poly_comms,
            opening_proof,
            shifted_opening_proof,
            poly_evals: ProofEvaluations {
                wires_evals,
                wire_sigma_evals,
                perm_next_eval,
            },
        })
    }

    pub fn prove<C, R>(
        prng: &mut R,
        circuit: &C,
        prove_key: &ProvingKey<Bls12_381>,
    ) -> Result<Proof<Bls12_381>, PlonkError>
    where
        C: Arithmetization<Fr>,
        R: CryptoRng + RngCore,
    {
        let domain = circuit.eval_domain_size();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        tokio::task::LocalSet::new().block_on(&rt, async {
            let network: NetworkConfig =
                serde_json::from_reader(File::open("config/network.json")?).unwrap();
            let num_slaves = network.slaves.len();

            let slaves = join_all(
                network
                    .slaves
                    .iter()
                    .map(|addr| async move { connect(addr).await.unwrap() }),
            )
            .await;

            Self::prove_async(&Context { slaves }, prng, circuit, prove_key).await
        })
    }
}

/// Private helper methods
impl Prover {
    // #[inline]
    // fn ifft(
    //     kernel: &SingleMultiexpKernel,
    //     domain: &Radix2EvaluationDomain<Fr>,
    //     coeffs: &mut Vec<Fr>,
    // ) {
    //     coeffs.resize(domain.size(), Fr::zero());
    //     kernel
    //         .radix_fft(coeffs, &domain.group_gen_inv, domain.log_size_of_group)
    //         .unwrap();
    //     coeffs.iter_mut().for_each(|val| *val *= domain.size_inv);
    // }

    // #[inline]
    // fn coset_fft(
    //     kernel: &SingleMultiexpKernel,
    //     domain: &Radix2EvaluationDomain<Fr>,
    //     coeffs: &mut Vec<Fr>,
    // ) {
    //     Radix2EvaluationDomain::distribute_powers(
    //         coeffs,
    //         Fr::multiplicative_generator(),
    //     );
    //     coeffs.resize(domain.size(), Fr::zero());
    //     kernel
    //         .radix_fft(coeffs, &domain.group_gen, domain.log_size_of_group)
    //         .unwrap();
    // }

    // #[inline]
    // fn coset_ifft(
    //     kernel: &SingleMultiexpKernel,
    //     domain: &Radix2EvaluationDomain<Fr>,
    //     coeffs: &mut Vec<Fr>,
    // ) {
    //     coeffs.resize(domain.size(), Fr::zero());
    //     kernel
    //         .radix_fft(coeffs, &domain.group_gen_inv, domain.log_size_of_group)
    //         .unwrap();
    //     coeffs.iter_mut().for_each(|val| *val *= domain.size_inv);
    //     Radix2EvaluationDomain::distribute_powers(
    //         coeffs,
    //         Fr::multiplicative_generator().inverse().unwrap(),
    //     );
    // }

    #[inline]
    fn commit_polynomial(
        // context: &mut MultiScalarMultContext,
        ck: &[G1Affine],
        poly: &[Fr],
    ) -> Commitment<Bls12_381> {
        let mut plain_coeffs = poly.iter().map(|s| s.into_repr()).collect::<Vec<_>>();

        plain_coeffs.resize(ck.len(), Fr::zero().into_repr());

        let commitment = VariableBaseMSM::multi_scalar_mul(&ck, &plain_coeffs);
        // let commitment =
        //     kernel.multiexp(&ck, &plain_coeffs).unwrap();
        // let commitment = multi_scalar_mult::<G1Affine>(context, ck.len(), unsafe {
        //     std::mem::transmute(plain_coeffs.as_slice())
        // })[0];

        Commitment(commitment.into())
    }
}

/// Merkle Tree height
pub const TREE_HEIGHT: u8 = 32;
/// Number of memberships proofs to be verified in the circuit
pub const NUM_MEMBERSHIP_PROOFS: usize = 1;

/// generate a gigantic circuit (with random, satisfiable wire assignments)
///
/// num_constraint = num_memebership_proof * (157 * tree_height + 149)
pub fn generate_circuit<R: Rng>(rng: &mut R) -> Result<PlonkCircuit<Fr>, PlonkError> {
    let mut leaves = vec![];
    let mut merkle_proofs = vec![];

    // sample leaves and insert into the merkle tree
    let mut mt = MerkleTree::new(TREE_HEIGHT).expect("Failed to initialize merkle tree");

    for _ in 0..NUM_MEMBERSHIP_PROOFS {
        let leaf = Fr::rand(rng);
        mt.push(leaf);
        leaves.push(leaf);
    }
    for uid in 0..NUM_MEMBERSHIP_PROOFS {
        merkle_proofs.push(
            AccMemberWitness::lookup_from_tree(&mt, uid as u64)
                .expect_ok()
                .expect("Failed to generate merkle proof")
                .1,
        );
    }
    let root = mt.commitment().root_value.to_scalar();

    // construct circuit constraining membership proof check
    let mut circuit = PlonkCircuit::new();
    // add root as a public input
    let root_var = circuit.create_public_variable(root)?;
    for (uid, proof) in merkle_proofs.iter().enumerate() {
        let leaf_var = circuit.create_variable(leaves[uid])?;
        let proof_var = AccMemberWitnessVar::new(&mut circuit, proof)?;
        let acc_elem_var = AccElemVars {
            uid: proof_var.uid,
            elem: leaf_var,
        };

        let claimed_root_var = circuit.compute_merkle_root(acc_elem_var, &proof_var.merkle_path)?;

        // enforce matching merkle root
        circuit.equal_gate(root_var, claimed_root_var)?;
    }

    // sanity check: the circuit must be satisfied.
    assert!(circuit.check_circuit_satisfiability(&[root]).is_ok());
    circuit.finalize_for_arithmetization()?;

    Ok(circuit)
}

#[test]
fn test_plonk() {
    let mut rng = rand::thread_rng();

    let circuit = generate_circuit(&mut rng).unwrap();
    let srs =
        PlonkKzgSnark::<Bls12_381>::universal_setup(circuit.srs_size().unwrap(), &mut rng).unwrap();
    let (pk, vk) = PlonkKzgSnark::<Bls12_381>::preprocess(&srs, &circuit).unwrap();

    // verify the proof against the public inputs.
    let proof = Prover::prove(&mut rng, &circuit, &pk).unwrap();
    let public_inputs = circuit.public_input().unwrap();
    assert!(
        PlonkKzgSnark::<Bls12_381>::verify::<StandardTranscript>(&vk, &public_inputs, &proof,)
            .is_ok()
    );
}
