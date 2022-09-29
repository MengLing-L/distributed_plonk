use ark_bls12_381::Fr;
use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::rand::thread_rng;
use ark_std::UniformRand;

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

fn fft(domain: &Radix2EvaluationDomain<Fr>, coeffs: &Vec<Fr>) -> Vec<Fr> {
    let r = 1 << (domain.log_size_of_group >> 1);
    let c = domain.size() / r;
    let r_domain = Radix2EvaluationDomain::<Fr>::new(r).unwrap();
    let c_domain = Radix2EvaluationDomain::<Fr>::new(c).unwrap();

    let mut coeffs = coeffs.clone();
    coeffs.resize(domain.size(), Fr::default());
    let mut t = transpose(coeffs.chunks(r).map(|i| i.to_vec()).collect::<Vec<_>>());
    t.iter_mut().enumerate().for_each(|(i, group)| {
        c_domain.fft_in_place(group);
        group
            .iter_mut()
            .enumerate()
            .for_each(|(j, u)| *u *= domain.group_gen.pow([(i * j) as u64]))
    });
    let mut groups = transpose(t);
    groups
        .iter_mut()
        .for_each(|group| r_domain.fft_in_place(group));
    transpose(groups).concat()
}

fn ifft(domain: &Radix2EvaluationDomain<Fr>, coeffs: &Vec<Fr>) -> Vec<Fr> {
    let r = 1 << (domain.log_size_of_group >> 1);
    let c = domain.size() / r;
    let r_domain = Radix2EvaluationDomain::<Fr>::new(r).unwrap();
    let c_domain = Radix2EvaluationDomain::<Fr>::new(c).unwrap();

    let mut coeffs = coeffs.clone();
    coeffs.resize(domain.size(), Fr::default());
    let mut t = transpose(coeffs.chunks(r).map(|i| i.to_vec()).collect::<Vec<_>>());
    t.iter_mut().enumerate().for_each(|(i, group)| {
        c_domain.ifft_in_place(group);
        group
            .iter_mut()
            .enumerate()
            .for_each(|(j, u)| *u *= domain.group_gen_inv.pow([(i * j) as u64]))
    });
    let mut groups = transpose(t);
    groups
        .iter_mut()
        .for_each(|group| r_domain.ifft_in_place(group));
    transpose(groups).concat()
}

fn coset_fft(domain: &Radix2EvaluationDomain<Fr>, coeffs: &Vec<Fr>) -> Vec<Fr> {
    let mut coeffs = coeffs.clone();
    Radix2EvaluationDomain::distribute_powers(&mut coeffs, Fr::multiplicative_generator());
    fft(domain, &coeffs)
}

fn coset_ifft(domain: &Radix2EvaluationDomain<Fr>, coeffs: &Vec<Fr>) -> Vec<Fr> {
    let mut coeffs = ifft(domain, coeffs);
    Radix2EvaluationDomain::distribute_powers(
        &mut coeffs,
        Fr::multiplicative_generator().inverse().unwrap(),
    );
    coeffs
}

#[test]
fn test() {
    let rng = &mut thread_rng();

    let l = 512;

    let domain = Radix2EvaluationDomain::<Fr>::new(l).unwrap();
    let quot_domain = Radix2EvaluationDomain::<Fr>::new(l * 2).unwrap();

    let exps = (0..l).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
    let mut t = exps.clone();
    t.resize(l * 2, Fr::default());

    assert_eq!(fft(&domain, &exps), domain.fft(&exps));
    assert_eq!(ifft(&domain, &exps), domain.ifft(&exps));
    assert_eq!(coset_fft(&domain, &exps), domain.coset_fft(&exps));
    assert_eq!(coset_ifft(&domain, &exps), domain.coset_ifft(&exps));

    assert_eq!(quot_domain.coset_fft(&exps), quot_domain.coset_fft(&t));
    assert_eq!(quot_domain.fft(&exps), fft(&quot_domain, &exps));
    assert_eq!(domain.coset_ifft(&domain.coset_fft(&exps)), exps);
}
