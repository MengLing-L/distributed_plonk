@0x9663f4dd604afa35;

struct FftWorkload {
    rowStart @0 : UInt64;
    rowEnd @1 : UInt64;
    colStart @2 : UInt64;
    colEnd @3 : UInt64;
}

interface Plonk {
    init @0 (bases :Data, domainSize :UInt64, quotDomainSize :UInt64);
    varMsm @1 (scalars :Data) -> (result: Data);

    fftInit @2 (id: UInt64, workloads: List(FftWorkload), is_quot: Bool, is_inv: Bool, is_coset: Bool);
    fft1 @3 (id: UInt64, i: UInt64, v: Data);
    fft2Prepare @4 (id: UInt64);
    fft2 @5 (id: UInt64) -> (v: Data);
}

interface FftPeer {
    exchange @0 (id: UInt64, from: UInt64, v: Data);
}
