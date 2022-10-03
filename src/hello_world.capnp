@0x9663f4dd604afa35;

struct MsmWorkload {
    start @0 : UInt64;
    end @1 : UInt64;
}

struct FftWorkload {
    rowStart @0 : UInt64;
    rowEnd @1 : UInt64;
    colStart @2 : UInt64;
    colEnd @3 : UInt64;
}

interface PlonkSlave {
    init @0 (bases :List(Data), domainSize :UInt64, quotDomainSize :UInt64);
    varMsm @1 (workload: MsmWorkload, scalars :List(Data)) -> (result: Data);

    fftInit @2 (id: UInt64, workloads: List(FftWorkload), is_quot: Bool, is_inv: Bool, is_coset: Bool);
    fft1 @3 (id: UInt64, i: UInt64, v: List(Data));
    fft2Prepare @4 (id: UInt64);
    fft2 @5 (id: UInt64) -> (v: List(Data));

    round1 @6 (w: List(Data)) -> (c: Data);

    round3Step1AH @7 (q_a: List(Data), q_h: List(Data)) -> (v: List(Data));
    round3Step1O @8 (q_o: List(Data)) -> (v: List(Data));
    round3Step2MInit @9 ();
    round3Step2MRetrieve @10 (q_m: List(Data)) -> (v: List(Data));
    round3Step2E @11 (q_e: List(Data)) -> (v: List(Data));
    round3Step3Init @12 (beta: Data, gamma: Data, k: Data);
    round3Step3Retrieve @13 () -> (v: List(Data));
    round3Step4Init @14 (sigma: List(Data));
    round3Step4Retrieve @15 () -> (v: List(Data));
    round3Step5 @16 (t: List(Data)) -> (c: Data);

    round4Type1 @17 (sigma: List(Data), zeta: Data) -> (v1: Data, v2: Data);
    round4Type2 @18 (zeta: Data) -> (w: Data);

    round5Step1 @19 () -> (v1: List(Data), v2: Data, v3: Data);
    round5Step2MInit @20 ();
    round5Step2MRetrieve @21 () -> (v: List(Data));
    round5Step2E @22 () -> (v: List(Data));
    round5Step3 @23 (v: Data) -> (v: List(Data));
}

interface PlonkPeer {
    fftExchange @0 (id: UInt64, from: UInt64, v: List(Data));

    round3Step2MExchange @1 (w: List(Data));
    round3Step2EExchange @2 (w: List(Data));
}
