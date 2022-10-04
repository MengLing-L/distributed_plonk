#![feature(int_roundings)]

use std::{error::Error, fs::File, net::ToSocketAddrs};

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::msm::VariableBaseMSM;
use ark_ff::{Fp256, FromBytes, PrimeField, ToBytes, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::{future::join_all, AsyncReadExt, TryFutureExt};
use hello_world::{
    config::NetworkConfig,
    hello_world_capnp::{plonk_slave},
    utils::{deserialize, serialize, FftWorkload, MsmWorkload},
};

use jf_plonk::prelude::*;
use jf_primitives::{
    circuit::merkle_tree::{AccElemVars, AccMemberWitnessVar, MerkleTreeGadget},
    merkle_tree::{AccMemberWitness, MerkleTree},
};
use rand::{distributions::Standard, thread_rng, Rng};

use ark_ec::AffineCurve;

use jf_plonk::{
    circuit::{gates::Gate, Arithmetization, GateId, Variable, WireId},
    constants::GATE_WIDTH,
    errors::{CircuitError, SnarkError},
    prelude::{
        Circuit, PlonkCircuit, PlonkError, Proof, ProofEvaluations, ProvingKey, VerifyingKey,
    },
    proof_system::{PlonkKzgSnark, Snark},
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

use ark_bls12_381::{Bls12_381, Fq, FrParameters};
use ark_ff::{FftField, FftParameters, Field, One};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Polynomial, Radix2EvaluationDomain, UVPolynomial,
};
use ark_std::{
    format,
    ops::Mul,
    println,
    rand::{CryptoRng, RngCore},
    time::Instant,
    vec,
    vec::Vec,
};

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

/// A Plonk IOP
pub struct Prover {}

impl Prover {
    pub async fn prove<C, R>(
        prng: &mut R,
        circuit: &C,
        prove_key: &ProvingKey<'_, Bls12_381>,
    ) -> Result<Proof<Bls12_381>, PlonkError>
    where
        C: Arithmetization<Fr>,
        R: CryptoRng + RngCore,
    {
        let network: NetworkConfig =
            serde_json::from_reader(File::open("config/network.json")?).unwrap();
        let num_slaves = network.slaves.len();
        let n = circuit.eval_domain_size()?;
        let num_wire_types = circuit.num_wire_types();

        let mut ck = prove_key.commit_key.powers_of_g.to_vec();
        ck.resize(((ck.len() + 31) >> 5) << 5, G1Affine::zero());
        let mut ck = &ck;

        if n == 1 {
            return Err(CircuitError::UnfinalizedCircuit.into());
        }
        if n < circuit.num_gates() {
            return Err(SnarkError::ParameterError(format!(
                "Domain size {} should be bigger than number of constraint {}",
                n,
                circuit.num_gates()
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
        if circuit.num_inputs() != prove_key.vk.num_inputs {
            return Err(SnarkError::ParameterError(format!(
                "circuit.num_inputs {} != prove_key.num_inputs {}",
                circuit.num_inputs(),
                prove_key.vk.num_inputs
            ))
            .into());
        }

        // Initialize transcript
        let mut transcript = FakeStandardTranscript::new(b"PlonkProof");
        let mut pub_input = circuit.public_input()?;

        transcript.append_vk_and_pub_input(&prove_key.vk, &pub_input)?;

        let domain = Radix2EvaluationDomain::<Fr>::new(n).ok_or(PlonkError::DomainCreationError)?;
        let quot_domain = Radix2EvaluationDomain::<Fr>::new((num_wire_types + 1) * (n + 1) + 1)
            .ok_or(PlonkError::DomainCreationError)?;
            let circuit = unsafe { &*(circuit as *const _ as *const FakePlonkCircuit<Fr>) };

        // Round 1
        tokio::task::LocalSet::new()
            .run_until(async move {
                let connections = &join_all(
                    network
                        .slaves
                        .iter()
                        .map(|addr| async move { connect(addr).await.unwrap() }),
                )
                .await;
                join_all(ck.chunks(ck.len() / num_slaves).zip(connections).map(
                    |(bases, connection)| async move {
                        init(connection, bases, domain.size(), quot_domain.size())
                            .await
                            .unwrap();
                    },
                ))
                .await;

                let r_d = 1 << (domain.log_size_of_group >> 1);
                let c_d = domain.size() / r_d;

                let workloads_d = &(0..num_slaves)
                    .map(|i| FftWorkload {
                        row_start: i * r_d / num_slaves,
                        row_end: (i + 1) * r_d / num_slaves,
                        col_start: i * c_d / num_slaves,
                        col_end: (i + 1) * c_d / num_slaves,
                    })
                    .collect::<Vec<_>>();

                let r_q = 1 << (quot_domain.log_size_of_group >> 1);
                let c_q = quot_domain.size() / r_q;

                let workloads_q = &(0..num_slaves)
                    .map(|i| FftWorkload {
                        row_start: i * r_q / num_slaves,
                        row_end: (i + 1) * r_q / num_slaves,
                        col_start: i * c_q / num_slaves,
                        col_end: (i + 1) * c_q / num_slaves,
                    })
                    .collect::<Vec<_>>();

                let now = Instant::now();
                let wire_polys = join_all(circuit
                    .wire_variables
                    .iter()
                    .take(num_wire_types)
                    .map(|wire_vars| async move {
                        let mut coeffs = wire_vars.iter().map(|&var| circuit.witness[var]).collect();
                        //domain.ifft_in_place(&mut coeffs);
                        // Self::ifft(&kernel, &domain, &mut coeffs);
                        Self::fft(connections, num_slaves, workloads_d, &domain, &mut coeffs, false, true, false).await.unwrap()
                        // DensePolynomial::rand(1, prng).mul_by_vanishing_poly(domain)
                        //     + DensePolynomial::from_coefficients_vec(coeffs)
                    })
                    .collect::<Vec<_>>()).await;
                let wire_polys = wire_polys.iter().map(|wire_vec| {
                    let mut coeffs = wire_vec.to_vec();
                    //domain.ifft_in_place(&mut coeffs);
                    // Self::ifft(&kernel, &domain, &mut coeffs);
                    DensePolynomial::rand(1, prng).mul_by_vanishing_poly(domain)
                         + DensePolynomial::from_coefficients_vec(coeffs)
                }).collect::<Vec<_>>();


                let wires_poly_comms = join_all(wire_polys.iter().map(|poly| async move {
                    Self::commit_polynomial(connections, &num_slaves, ck, poly)
                        .await
                        .unwrap()
                }))
                .await;
                transcript.append_commitments(b"witness_poly_comms", &wires_poly_comms);
                // println!("Elapsed: {:.2?}", now.elapsed());

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
                            b *= tmp
                                + beta * circuit.extended_id_permutation[perm_i * n + perm_j];
                        }
                        product_vec.push(product_vec[j] * a / b);
                    }
                    //domain.ifft_in_place(&mut product_vec);
                    product_vec = Self::fft(connections, num_slaves, workloads_d, &domain,&mut product_vec,false, true, false).await.unwrap();
                    //Self::ifft(&kernel, &domain, &mut product_vec);
                    DensePolynomial::rand(2, prng).mul_by_vanishing_poly(domain)
                        + DensePolynomial::from_coefficients_vec(product_vec)
                };
                let prod_perm_poly_comm =
                    Self::commit_polynomial(connections, &num_slaves, ck, &permutation_poly)
                        .await
                        .unwrap();

                transcript.append_commitment(b"perm_poly_comms", &prod_perm_poly_comm)?;
                // println!("Elapsed: {:.2?}", now.elapsed());

                // Round 3
                let now = Instant::now();
                let alpha = transcript.get_and_append_challenge::<Bls12_381>(b"alpha")?;
                let alpha_square_div_n = alpha.square() / Fr::from(n as u64);
                let quotient_poly = {
                    let m = quot_domain.size();
                    let mut eval_points = vec![Fr::multiplicative_generator()];
                    for i in 1..m {
                        eval_points.push(eval_points[i - 1] * quot_domain.group_gen);
                    }
                    let domain_size_ratio = m / n;
                    // Compute 1/Z_H(w^i).
                    let z_h_inv = (0..domain_size_ratio)
                        .into_iter()
                        .map(|i| {
                            (eval_points[i].pow([n as u64]) - Fr::one())
                                .inverse()
                                .unwrap()
                        })
                        .collect::<Vec<_>>();

                    // Compute coset evaluations.
                    let selectors_coset_fft = join_all(prove_key
                        .selectors
                        .iter()
                        .map(|poly| async move{
                            let mut coeffs = poly.coeffs().to_vec();
                            //quot_domain.coset_fft_in_place(&mut coeffs);
                            coeffs = Self::fft(connections, num_slaves, workloads_q, &quot_domain, &mut coeffs,true, false, true).await.unwrap();
                            // Self::coset_fft(&kernel, &quot_domain, &mut coeffs);
                            coeffs
                        })
                        .collect::<Vec<_>>()).await;
                    let sigmas_coset_fft = join_all(prove_key
                        .sigmas
                        .iter()
                        .map(|poly| async move{
                            let mut coeffs = poly.coeffs().to_vec();
                            //quot_domain.coset_fft_in_place(&mut coeffs);
                            coeffs = Self::fft(connections, num_slaves, workloads_q, &quot_domain, &mut coeffs,true, false, true).await.unwrap();
                            // Self::coset_fft(&kernel, &quot_domain, &mut coeffs);
                            coeffs
                        })
                        .collect::<Vec<_>>()).await;

                    let wire_polys_coset_fft = join_all(wire_polys
                        .iter()
                        .map(|poly| async move{
                            let mut coeffs = poly.coeffs().to_vec();
                            //quot_domain.coset_fft_in_place(&mut coeffs);
                            coeffs = Self::fft(connections, num_slaves, workloads_q, &quot_domain, &mut coeffs,true, false, true).await.unwrap();
                            // Self::coset_fft(&kernel, &quot_domain, &mut coeffs);
                            coeffs
                        })
                        .collect::<Vec<_>>()).await;
                    // TODO: (binyi) we can also compute below in parallel with
                    // `wire_polys_coset_fft`.
                    let prod_perm_poly_coset_fft = {
                        let mut coeffs = permutation_poly.coeffs().to_vec();
                        //quot_domain.coset_fft_in_place(&mut coeffs);
                        coeffs = Self::fft(connections, num_slaves, workloads_q, &quot_domain, &mut coeffs,true, false, true).await.unwrap();
                        // Self::coset_fft(&kernel, &quot_domain, &mut coeffs);
                        coeffs
                    };
                    let pub_input_poly_coset_fft = {
                        //domain.ifft_in_place(&mut pub_input);
                        pub_input = Self::fft(connections, num_slaves, workloads_d, &domain,&mut pub_input,false, true, false).await.unwrap();
                        //quot_domain.coset_fft_in_place(&mut pub_input);
                        pub_input = Self::fft(connections, num_slaves, workloads_q, &quot_domain, &mut pub_input,true, false, true).await.unwrap();
                        // Self::ifft(&kernel, &domain, &mut pub_input);
                        // Self::coset_fft(&kernel, &quot_domain, &mut pub_input);
                        pub_input
                    };

                    // Compute coset evaluations of the quotient polynomial.
                    let mut quot_poly_coset_evals: Vec<_> = (0..m)
                        .into_iter()
                        .map(|i| {
                            let eval_point = eval_points[i];

                            z_h_inv[i % domain_size_ratio]
                                * ({
                                    // Selectors
                                    // The order: q_lc, q_mul, q_hash, q_o, q_c, q_ecc
                                    // TODO: (binyi) get the order from a function.
                                    let q_lc: Vec<_> = (0..GATE_WIDTH)
                                        .map(|j| selectors_coset_fft[j][i])
                                        .collect();
                                    let q_mul: Vec<_> = (GATE_WIDTH..GATE_WIDTH + 2)
                                        .map(|j| selectors_coset_fft[j][i])
                                        .collect();
                                    let q_hash: Vec<_> = (GATE_WIDTH + 2..2 * GATE_WIDTH + 2)
                                        .map(|j| selectors_coset_fft[j][i])
                                        .collect();
                                    let q_o = selectors_coset_fft[2 * GATE_WIDTH + 2][i];
                                    let q_c = selectors_coset_fft[2 * GATE_WIDTH + 3][i];
                                    let q_ecc = selectors_coset_fft[2 * GATE_WIDTH + 4][i];

                                    let a = wire_polys_coset_fft[0][i];
                                    let b = wire_polys_coset_fft[1][i];
                                    let c = wire_polys_coset_fft[2][i];
                                    let d = wire_polys_coset_fft[3][i];
                                    let e = wire_polys_coset_fft[4][i];
                                    let ab = a * b;
                                    let cd = c * d;
                                    q_c + pub_input_poly_coset_fft[i]
                                        + q_lc[0] * a
                                        + q_lc[1] * b
                                        + q_lc[2] * c
                                        + q_lc[3] * d
                                        + q_mul[0] * ab
                                        + q_mul[1] * cd
                                        + q_ecc * ab * cd * e
                                        + q_hash[0] * a.square().square() * a
                                        + q_hash[1] * b.square().square() * b
                                        + q_hash[2] * c.square().square() * c
                                        + q_hash[3] * d.square().square() * d
                                        - q_o * e
                                } + {
                                    // The check that:
                                    //   \prod_i [w_i(X) + beta * k_i * X + gamma] * z(X)
                                    // - \prod_i [w_i(X) + beta * sigma_i(X) + gamma] * z(wX) = 0
                                    // on the vanishing set.
                                    // Delay the division of Z_H(X).
                                    //
                                    // Extended permutation values
                                    let mut acc1 = prod_perm_poly_coset_fft[i];
                                    let mut acc2 =
                                        prod_perm_poly_coset_fft[(i + domain_size_ratio) % m];
                                    for j in 0..num_wire_types {
                                        let t = wire_polys_coset_fft[j][i] + gamma;
                                        acc1 *= t + prove_key.vk.k[j] * eval_point * beta;
                                        acc2 *= t + sigmas_coset_fft[j][i] * beta;
                                    }
                                    alpha * (acc1 - acc2)
                                })
                                + {
                                    // The check that z(x) = 1 at point 1.
                                    // (z(x)-1) * L1(x) * alpha^2 / Z_H(x) = (z(x)-1) * alpha^2 / (n * (x -
                                    // 1))
                                    alpha_square_div_n * (prod_perm_poly_coset_fft[i] - Fr::one())
                                        / (eval_point - Fr::one())
                                }
                        })
                        .collect();

                    // Compute the coefficient form of the quotient polynomial
                    quot_domain.coset_ifft_in_place(&mut quot_poly_coset_evals);
                    // Self::coset_ifft(&kernel, &quot_domain, &mut quot_poly_coset_evals);
                    DensePolynomial::from_coefficients_vec(quot_poly_coset_evals)
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
                        .map(|coeffs| DensePolynomial::<Fr>::from_coefficients_slice(coeffs))
                        .collect::<Vec<_>>()
                };
                let split_quot_poly_comms =
                    join_all(split_quot_polys.iter().map(|poly| async move {
                        Self::commit_polynomial(connections, &num_slaves, ck, poly)
                            .await
                            .unwrap()
                    }))
                    .await;
                transcript.append_commitments(b"quot_poly_comms", &split_quot_poly_comms)?;

                // let mut split_quot_poly_comms= vec![];
                // async{
                //     split_quot_poly_comms = Self::commit_polynomial_poly(&connections, &num_slaves,&ck, &split_quot_polys).await.unwrap();
                //     transcript.append_commitments(b"witness_poly_comms", &split_quot_poly_comms);
                // }.await;
                // println!("Elapsed: {:.2?}", now.elapsed());

                // Round 4
                let now = Instant::now();
                let zeta = transcript.get_and_append_challenge::<Bls12_381>(b"zeta")?;
                let wires_evals = wire_polys
                    .iter()
                    .map(|poly| poly.evaluate(&zeta))
                    .collect::<Vec<_>>();
                let wire_sigma_evals = prove_key
                    .sigmas
                    .iter()
                    .take(num_wire_types - 1)
                    .map(|poly| poly.evaluate(&zeta))
                    .collect::<Vec<_>>();
                let perm_next_eval = permutation_poly.evaluate(&(zeta * domain.group_gen));
                transcript.append_proof_evaluations::<Bls12_381>(
                    &wires_evals,
                    &wire_sigma_evals,
                    &perm_next_eval,
                )?;
                // println!("Elapsed: {:.2?}", now.elapsed());

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
                    let num_wire_types = wires_evals.len();
                    let coeff = -wires_evals
                        .iter()
                        .take(num_wire_types - 1)
                        .zip(&wire_sigma_evals)
                        .fold(
                            alpha * beta * perm_next_eval,
                            |acc, (&wire_eval, &sigma_eval)| {
                                acc * (wire_eval + beta * sigma_eval + gamma)
                            },
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

                    let witness_poly = {
                        let mut quotient = vec![Fr::zero(); batch_poly.degree()];
                        let mut remainder = batch_poly.clone();
                        while !remainder.is_zero() && remainder.degree() >= 1 {
                            let cur_q_coeff = *remainder.coeffs.last().unwrap();
                            let cur_q_degree = remainder.degree() - 1;
                            quotient[cur_q_degree] = cur_q_coeff;

                            remainder[cur_q_degree] += &(cur_q_coeff * zeta);
                            remainder[cur_q_degree + 1] -= &cur_q_coeff;
                            while let Some(true) = remainder.coeffs.last().map(|c| c.is_zero()) {
                                remainder.coeffs.pop();
                            }
                        }
                        quotient
                    };
                    Self::commit_polynomial(connections, &num_slaves, ck, &witness_poly)
                        .await
                        .unwrap()
                };

                let shifted_opening_proof = {
                    let witness_poly = {
                        let mut quotient = vec![Fr::zero(); permutation_poly.degree()];
                        let mut remainder = permutation_poly.clone();
                        while !remainder.is_zero() && remainder.degree() >= 1 {
                            let cur_q_coeff = *remainder.coeffs.last().unwrap();
                            let cur_q_degree = remainder.degree() - 1;
                            quotient[cur_q_degree] = cur_q_coeff;

                            remainder[cur_q_degree] += &(cur_q_coeff * domain.group_gen * zeta);
                            remainder[cur_q_degree + 1] -= &cur_q_coeff;
                            while let Some(true) = remainder.coeffs.last().map(|c| c.is_zero()) {
                                remainder.coeffs.pop();
                            }
                        }
                        quotient
                    };
                    Self::commit_polynomial(connections, &num_slaves, ck, &witness_poly)
                        .await
                        .unwrap()
                };
                // println!("Elapsed: {:.2?}", now.elapsed());

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
            })
            .await
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

    #[inline]
    async fn fft(
        connections: &Vec<plonk_slave::Client>,
        num_slaves: usize,
        workloads: &[FftWorkload],
        domain: &Radix2EvaluationDomain<Fr>,
        coeffs: &mut Vec<Fr>,
        is_quot: bool,
        is_inv: bool,
        is_coset: bool
    )-> Result<Vec<Fr>, Box<dyn Error>>  {
        let rng = &mut thread_rng();
        let id: u64 = rng.gen();
        let r = 1 << (domain.log_size_of_group >> 1);
        let c = domain.size() / r;
        coeffs.resize(domain.size(), Default::default());

        join_all(connections.iter().map(|connection| async move {
            fft_init(connection, id, workloads, is_quot, is_inv, is_coset)
                .await
                .unwrap()
        }))
        .await;
        let t = &transpose(coeffs.chunks(r).map(|i| i.to_vec()).collect::<Vec<_>>());
        //println!("{:?}",t);
        join_all(connections.iter().zip(t.chunks(r / num_slaves)).map(
            |(connection, tt)| async move {
                join_all(
                    tt.iter()
                        .enumerate()
                        .map(|(j, v)| async move { fft1(connection, id, j, v).await.unwrap() }),
                )
                .await
            },
        ))
        .await;
        join_all(
            connections
                .iter()
                .map(|connection| async move { fft2_prepare(connection, id).await.unwrap() }),
        )
        .await;
        let mut u = vec![vec![]; c];
        let t = join_all(
            connections
                .iter()
                .map(|connection| async move { fft2(connection, id).await.unwrap() }),
        )
        .await;
        for i in 0..num_slaves {
            for (j, v) in t[i].chunks(r).enumerate() {
                u[i * c / num_slaves + j] = v.to_vec();
            }
        }

        Ok(transpose(u).concat())
    }
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
    async fn commit_polynomial(
        connections: &Vec<plonk_slave::Client>,
        num_slaves: &usize,
        bases: &[G1Affine],
        poly: &[Fr],
    ) -> Result<Commitment<Bls12_381>, Box<dyn Error>> {
        let mut plain_coeffs = poly.iter().map(|s| s.into_repr()).collect::<Vec<_>>();

        plain_coeffs.resize(bases.len(), Fr::zero().into_repr());

        //let commitment = VariableBaseMSM::multi_scalar_mul(&bases, &plain_coeffs);
        // join_all(
        //     bases
        //         .chunks(bases.len() / num_slaves)
        //         .zip(connections)
        //         .map(|(bases, connection)| async move {
        //             init(connection, bases, 0, 0).await.unwrap();
        //         }),
        // )
        // .await;

        // let commitment =
        //     join_all(plain_coeffs.chunks(plain_coeffs.len() / num_slaves).zip(connections).map(
        //         |(exps, connection)| async move { msm(&connection, exps).await.unwrap() },
        //     ))
        //     .await
        //     .into_iter()
        //     .reduce(|a, b| a + b)
        //     .unwrap();
        // let commitment =
        //     kernel.multiexp(&ck, &plain_coeffs).unwrap();
        // let commitment = multi_scalar_mult::<G1Affine>(context, ck.len(), unsafe {
        //     std::mem::transmute(plain_coeffs.as_slice())
        // })[0];

        let commitment = join_all(
            plain_coeffs.chunks(plain_coeffs.len() / num_slaves)
                .enumerate()
                .map(|(i, scalars)| {
                    (
                        MsmWorkload {
                            start: i * plain_coeffs.len() / num_slaves,
                            end: (i + 1) * plain_coeffs.len() / num_slaves,
                        },
                        scalars,
                    )
                })
                .zip(connections)
                .map(|((workload, scalars), connection)| async move {
                    msm(&connection, &workload, scalars).await.unwrap()
                }),
        )
        .await
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap();

        Ok(Commitment(commitment.into()))
    }

    // #[inline]
    // async fn commit_polynomial_poly(
    //     connections: &Vec<plonk::Client>,
    //     num_slaves: &usize,
    //     bases: &[G1Affine],
    //     wire_polys: &Vec<DensePolynomial<Fp256<FrParameters>>>,
    // ) -> Result<Vec<Commitment<Bls12_381>>, Box<dyn Error>> {
    //     let res = join_all(wire_polys
    //         .iter()
    //         .map(|poly| async move {
    //                 let mut plain_coeffs = poly.iter().map(|s| s.into_repr()).collect::<Vec<_>>();

    //                 plain_coeffs.resize(bases.len(), Fr::zero().into_repr());

    //                 let commitment =
    //                     join_all(plain_coeffs.chunks(plain_coeffs.len() / num_slaves).zip(connections).map(
    //                         |(exps, connection)| async move { msm(&connection, exps).await.unwrap() },
    //                     ))
    //                     .await
    //                     .into_iter()
    //                     .reduce(|a, b| a + b)
    //                     .unwrap();
    //                 Commitment(commitment.into())
    //         })
    //         .collect::<Vec<_>>()).await;

    //     // let commitment =
    //     //     kernel.multiexp(&ck, &plain_coeffs).unwrap();
    //     // let commitment = multi_scalar_mult::<G1Affine>(context, ck.len(), unsafe {
    //     //     std::mem::transmute(plain_coeffs.as_slice())
    //     // })[0];

    //     Ok(res)
    // }
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

async fn connect<A: tokio::net::ToSocketAddrs>(addr: A) -> Result<plonk_slave::Client, Box<dyn Error>> {
    let stream = tokio::net::TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();
    let rpc_network = Box::new(twoparty::VatNetwork::new(
        reader,
        writer,
        rpc_twoparty_capnp::Side::Client,
        Default::default(),
    ));
    let mut rpc_system = RpcSystem::new(rpc_network, None);
    let client: plonk_slave::Client = rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);
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

#[tokio::main(flavor = "current_thread")]
pub async fn main() -> Result<(), Box<dyn Error>> {
    let network: NetworkConfig = serde_json::from_reader(File::open("config/network.json")?)?;
    let num_slaves = network.slaves.len();

    let rng = &mut thread_rng();

    let l = 1 << 20;

    let mut bases = (0..1 << 11)
        .map(|_| G1Projective::rand(rng).into())
        .collect::<Vec<G1Affine>>();
    // Sprinkle in some infinity points
    bases[3] = G1Projective::zero().into();
    while bases.len() < l {
        bases.append(&mut bases.clone());
    }

    // let exps = (0..l)
    //     .map(|_| Fr::rand(rng).into_repr())
    //     .collect::<Vec<_>>();
    // println!("{}", VariableBaseMSM::multi_scalar_mul(&bases, &exps));
    let domain = Radix2EvaluationDomain::<Fr>::new(128).unwrap();
    let quot_domain = Radix2EvaluationDomain::<Fr>::new(1024).unwrap();

    tokio::task::LocalSet::new()
        .run_until(async move {
            let connections = join_all(
                network
                    .slaves
                    .iter()
                    .map(|addr| async move { connect(addr).await.unwrap() }),
            )
            .await;

            join_all(bases.chunks(l / connections.len()).zip(&connections).map(
                |(bases, connection)| async move {
                    init(connection, bases, domain.size(), quot_domain.size())
                        .await
                        .unwrap();
                },
            ))
            .await;

            // let r =
            //     join_all(exps.chunks(l / connections.len()).zip(&connections).map(
            //         |(exps, connection)| async move { msm(&connection, exps).await.unwrap() },
            //     ))
            //     .await
            //     .into_iter()
            //     .reduce(|a, b| a + b)
            //     .unwrap();

            // println!("{}", r);

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

                for is_inv in [false, true] {
                    for is_coset in [false, true] {
                        let id: u64 = rng.gen();

                        let coeffs = (0..domain.size())
                            .map(|_| Fr::rand(rng))
                            .collect::<Vec<_>>();

                        join_all(connections.iter().map(|connection| async move {
                            fft_init(connection, id, workloads, is_quot, is_inv, is_coset)
                                .await
                                .unwrap()
                        }))
                        .await;
                        let t =
                            &transpose(coeffs.chunks(r).map(|i| i.to_vec()).collect::<Vec<_>>());

                        join_all(connections.iter().zip(t.chunks(r / num_slaves)).map(
                            |(connection, tt)| async move {
                                join_all(tt.iter().enumerate().map(|(j, v)| async move {
                                    fft1(connection, id, j, v).await.unwrap()
                                }))
                                .await
                            },
                        ))
                        .await;
                        join_all(connections.iter().map(|connection| async move {
                            fft2_prepare(connection, id).await.unwrap()
                        }))
                        .await;
                        let mut u = vec![vec![]; c];
                        let t =
                            join_all(connections.iter().map(|connection| async move {
                                fft2(connection, id).await.unwrap()
                            }))
                            .await;
                        for i in 0..num_slaves {
                            for (j, v) in t[i].chunks(r).enumerate() {
                                u[i * c / num_slaves + j] = v.to_vec();
                            }
                        }

                        assert_eq!(
                            transpose(u).concat(),
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

/// Merkle Tree height
pub const TREE_HEIGHT: u8 = 32;
/// Number of memberships proofs to be verified in the circuit
pub const NUM_MEMBERSHIP_PROOFS: usize = 50;

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
fn test2() {
    let mut rng = rand::thread_rng();

    let circuit = generate_circuit(&mut rng).unwrap();
    let srs =
        PlonkKzgSnark::<Bls12_381>::universal_setup(circuit.srs_size().unwrap(), &mut rng).unwrap();
    let (pk, vk) = PlonkKzgSnark::<Bls12_381>::preprocess(&srs, &circuit).unwrap();

    // verify the proof against the public inputs.
    tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
    .unwrap()
    .block_on(async {
        let proof = Prover::prove(&mut rng, &circuit, &pk).await.unwrap();
        let public_inputs = circuit.public_input().unwrap();
        assert!(
            PlonkKzgSnark::<Bls12_381>::verify::<StandardTranscript>(&vk, &public_inputs, &proof,)
                .is_ok()
        );
    })
}