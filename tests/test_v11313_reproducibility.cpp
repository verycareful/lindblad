// Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
// SPDX-License-Identifier: LicenseRef-Lindblad-2.3
//
// This file is part of the Lindblad Quantum Computing Framework and is
// licensed under the Lindblad Software License Agreement, Version 2.3. The
// full text is in the LICENSE file at the root of the repository. Free for
// non-commercial and academic use; commercial use requires a separate
// Commercial License Agreement with the Author.

// 1.1.31.3 - the same input and seed give the same bits, whatever the heap
// layout.
//
// V11311SvdSettings.SilentCountsEveryRescueItDoesNotReport failed once in CI
// (GCC 13): one qutrit MPS chain, run twice in one process, made 7 Jacobi
// rescues the first time and 6 the second. The chain was not a function of its
// input. autonne's BDC kernel, built by GCC for a target with FMA, factorised
// one degenerate block in different ways depending on where the heap placed
// its buffers: GCC versions loops it cannot prove alias-free into a scalar and
// a vector copy and contracts multiply-adds differently in each, and which copy
// runs depends on how far apart two buffers are (shek014/autonne#20). The
// build now compiles autonne with -ffp-contract=off.
//
// Each test changes the heap layout between calls by allocating a block of
// pseudo-random size before each one, and requires every call to give the same
// bits. The block below is the one that factorised three ways: 9x27, column
// major, nine singular values equal to 1/3 within a few ulps, captured from
// the chain's step-7 gate. It is written as hex-float literals, so it is
// bit-exact.

#include <gtest/gtest.h>

#include "lindblad/constants.hpp"
#include "lindblad/detail/svd_truncate.hpp"
#include "lindblad/qudit/qudit_gates.hpp"
#include "lindblad/qudit/qudit_mps.hpp"
#include "lindblad/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

using namespace lindblad;

namespace {

// clang-format off
const std::vector<Complex128> kDegenerateBlock = {
    {0x1.47a8346434b29p-4, 0x1.d8683b93915dap-9}, {-0x1.2360d1be81e67p-8, -0x1.7feccb7e022b4p-11},
    {0x1.0fe5a249b0dbdp-7, 0x1.1b3673045d47bp-6}, {-0x1.e35f304b6b35ep-8, 0x1.2e7c8c8bf6ac4p-5},
    {-0x1.06a32608fbe31p-6, -0x1.d0527aa694798p-7}, {0x1.0c0a879f7d92dp-8, 0x1.c4c1b174faeb4p-8},
    {0x1.3a40624a48652p-5, -0x1.4980821516109p-3}, {0x1.bc5ff1405c6d3p-6, -0x1.3d8b131b5d657p-7},
    {-0x1.08d205ff172c9p-10, -0x1.56adce2612291p-7}, {0x1.db800148c4ddfp-6, 0x1.1c4b78543b2efp-4},
    {0x1.f2a2d32477ca7p-5, -0x1.3f1636d0856e2p-8}, {-0x1.5e24a0243f9e0p-7, 0x1.a2364409ce738p-7},
    {-0x1.0ba007f505224p-5, 0x1.c8553d4a60246p-4}, {-0x1.1e79bd7203380p-9, 0x1.4ba75ec5b8f70p-5},
    {-0x1.c77b003d4606ep-8, -0x1.ebe3442c7440ap-6}, {-0x1.44dda7c764ff5p-5, 0x1.53d90baae4fe8p-5},
    {0x1.14f2c6e451dfbp-4, -0x1.b9d7ce5314924p-5}, {0x1.1674766563c83p-6, 0x1.f6c729b44f8adp-7},
    {0x1.4b2a4813faa06p-6, 0x1.58f844b4b2896p-5}, {-0x1.c18c31cfd9acap-5, 0x1.b9918e84d9233p-8},
    {0x1.9faaab8569ddfp-7, -0x1.0d192ad6cd4cfp-6}, {-0x1.ea5d58a5f67bcp-6, 0x1.6caf66f3981cdp-4},
    {0x1.603aacfd7bbf7p-5, -0x1.f010464f62e02p-6}, {-0x1.735861603ec2ap-8, -0x1.caba55e53365fp-6},
    {-0x1.045ba5a2658e1p-5, -0x1.f23a044948ea2p-11}, {-0x1.b89a95c1cac47p-4, 0x1.37ffcf3749ccep-4},
    {-0x1.33702cafb9db8p-6, 0x1.eda07cdb32962p-7}, {-0x1.5ad5753d2d369p-10, 0x1.2bc1be45649e6p-3},
    {-0x1.ca1c2a3d73788p-9, 0x1.f1c8a4eceaf60p-9}, {-0x1.18d04e8ef1d8fp-8, -0x1.75c372fe38962p-6},
    {0x1.4d665488bcb11p-4, -0x1.1aaa6a5b8f2e7p-4}, {-0x1.ad2fa717680a8p-12, 0x1.aa588d5c63fd0p-7},
    {-0x1.5e3cdfb80a3aep-7, -0x1.f23ba19c7c1b8p-10}, {-0x1.77cc277ded2fep-5, -0x1.11ea9ce76aa86p-5},
    {-0x1.5ba026d2ab357p-7, -0x1.85f59f4241d0ap-8}, {-0x1.fe89d278b38c0p-12, 0x1.25dd008487d25p-10},
    {0x1.6c4c96cdcb694p-9, 0x1.79ff7ef39431fp-7}, {-0x1.b4650f03c71bap-4, -0x1.b1b8b30cf53d5p-4},
    {-0x1.552fa2dfef6cbp-5, 0x1.baf26190833abp-7}, {-0x1.b4a96767591edp-8, 0x1.8f8cefd96a032p-7},
    {-0x1.a41525397e6dbp-4, 0x1.08ce80c72fb7ep-6}, {0x1.c6b01877f1524p-9, -0x1.873ee9e9b21a0p-11},
    {-0x1.936a2a306ebb8p-7, 0x1.14908fb34536ep-6}, {0x1.839e462268da6p-6, 0x1.ff66f12eb748bp-7},
    {0x1.72c323205da4ep-7, -0x1.0263800870504p-7}, {0x1.57f19db5975f9p-8, 0x1.d578b9157238fp-6},
    {0x1.d49fcfaa0df92p-7, 0x1.3ccbc39b7cca1p-9}, {0x1.8cd5464abc40dp-4, 0x1.f36f82a3c200fp-4},
    {0x1.24c14c7893306p-7, 0x1.3437323b333dcp-9}, {-0x1.5dd7776de24dep-5, -0x1.72f6c0e27406cp-5},
    {0x1.89205c3240a9ep-5, 0x1.8b42f8077011ap-5}, {-0x1.9099c4747a5d8p-7, 0x1.d5d233456e3a2p-6},
    {-0x1.dd8035f109212p-8, 0x1.7deef6a33ce14p-8}, {-0x1.737d0a806002cp-5, -0x1.1840dc80c37e4p-8},
    {0x1.5841851a3a450p-8, 0x1.4fc55df6dfc64p-7}, {-0x1.6383514c091a6p-5, 0x1.26510e13e40c7p-4},
    {-0x1.8d5db1be0f9f8p-10, -0x1.69bb5ffde6471p-5}, {-0x1.aaa1e16c55ce5p-7, 0x1.bf01b2e6306b0p-8},
    {-0x1.f333a6050c07ap-5, -0x1.d9bda24d9631ap-4}, {-0x1.d583485591547p-7, -0x1.6659991d5ebe6p-6},
    {0x1.31043d632de3ap-10, 0x1.5bbd268e74105p-6}, {-0x1.3dd265bf65f3dp-7, -0x1.8586b2ee71f76p-4},
    {0x1.714138c7ee916p-7, -0x1.ce2e0e1f5f72cp-7}, {0x1.2df197a423ae1p-9, 0x1.2ffcce75c1ab6p-7},
    {0x1.6ba14db7844c6p-7, 0x1.acf6010dc45eep-9}, {0x1.c636a1b5a0200p-6, -0x1.f0008ad9417d0p-6},
    {-0x1.3c65e921345c6p-8, 0x1.a7f7692f19e9dp-7}, {0x1.542ce32cf4210p-9, 0x1.f700a0679504ap-8},
    {0x1.31cdcc498f2fcp-5, 0x1.202a0cba5ae64p-8}, {-0x1.569677112b954p-7, 0x1.4c11deb193686p-7},
    {0x1.505405ae16454p-8, 0x1.df70c4eb5c051p-6}, {0x1.98ab5f5bb5b9dp-5, -0x1.61fadb405892dp-3},
    {0x1.d9e47a07b5c21p-9, 0x1.78d6f7a6e0c1cp-7}, {-0x1.eb69edfcc6848p-7, 0x1.4a38a354bf982p-8},
    {-0x1.e70bf6716d81ep-7, -0x1.2cd04b9574ae7p-4}, {-0x1.ef4c39fdc6706p-9, 0x1.1f463e1027c46p-5},
    {0x1.66beb06fb78c8p-7, 0x1.13561b51de45ep-7}, {0x1.69bd51bcf2e80p-5, 0x1.4cc13403d5218p-3},
    {-0x1.8380efb62ffa8p-9, 0x1.1635b41d9211ep-7}, {0x1.907ee4f5689d0p-8, -0x1.06c65acd30d7ep-6},
    {-0x1.737e5c483c52fp-6, 0x1.344030b44b768p-6}, {0x1.47a8346434b19p-4, 0x1.d8683b9391596p-9},
    {-0x1.2360d1be81ea6p-8, -0x1.7feccb7e02128p-11}, {0x1.0fe5a249b0dc6p-7, 0x1.1b3673045d4a1p-6},
    {0x1.242bf7c94c5a9p-5, -0x1.8baab9be7d9ccp-7}, {-0x1.16f47be5df436p-8, 0x1.5787f76a814bdp-6},
    {0x1.021402861062ep-8, -0x1.ca823c8ebf934p-8}, {0x1.ec26bbdd74a95p-4, 0x1.d193ae23862aep-4},
    {-0x1.52bf9d0189415p-8, 0x1.d039c4c98cec3p-6}, {0x1.3951e70348e8ep-7, 0x1.1d57f7f8db6f6p-8},
    {0x1.db800148c4e02p-6, 0x1.1c4b78543b2e1p-4}, {0x1.f2a2d32477caep-5, -0x1.3f1636d085528p-8},
    {-0x1.5e24a0243f9bcp-7, 0x1.a2364409ce758p-7}, {0x1.ce1a2b36e842fp-4, -0x1.c12028fa3272cp-6},
    {0x1.282c47a3929eap-5, -0x1.2ca453b3fe247p-6}, {-0x1.710d5f57870fep-6, 0x1.588eec63836adp-6},
    {-0x1.07c4936bcd004p-6, -0x1.c34417d3c4d97p-5}, {0x1.a6cbae9889a73p-7, 0x1.5e4e17e90f664p-4},
    {-0x1.64efccb64b57dp-6, 0x1.cdd1756ea43e2p-8}, {0x1.4b2a4813faa1fp-6, 0x1.58f844b4b288fp-5},
    {-0x1.c18c31cfd9abfp-5, 0x1.b9918e84d9434p-8}, {0x1.9faaab8569df9p-7, -0x1.0d192ad6cd4bcp-6},
    {0x1.791f48406f104p-4, -0x1.30b3bcc6a885dp-6}, {-0x1.86ea9b181ac54p-5, -0x1.6a0c0dee2477dp-6},
    {-0x1.5eda11ff034f6p-6, 0x1.35c335b9a8fb5p-6}, {0x1.11d7765c0cdf3p-6, -0x1.bb2b21ec9a2ecp-6},
    {-0x1.8f2dc9c767173p-7, -0x1.0cc970bbb0d35p-3}, {-0x1.e038958d44ba1p-9, -0x1.85a7efbd20dcdp-6},
    {-0x1.5ad5753d2d4adp-10, 0x1.2bc1be45649e2p-3}, {-0x1.ca1c2a3d73773p-9, 0x1.f1c8a4eceb19bp-9},
    {-0x1.18d04e8ef1d52p-8, -0x1.75c372fe38944p-6}, {-0x1.9b7edaec9ce09p-4, -0x1.26ccb16282e9fp-5},
    {0x1.77eeb30f84d5fp-7, -0x1.931d9315244e3p-8}, {0x1.e4bc00dc077a6p-9, 0x1.4e74524c8bcb6p-7},
    {0x1.a91e0a91ebbcbp-5, -0x1.78fbdec4254b8p-6}, {0x1.56ab91fa19c23p-7, -0x1.97201df9350f6p-8},
    {-0x1.7d59ff87da2f2p-11, -0x1.017768d91411cp-10}, {0x1.6c4c96cdcb557p-9, 0x1.79ff7ef39429cp-7},
    {-0x1.b4650f03c71aep-4, -0x1.b1b8b30cf53f6p-4}, {-0x1.552fa2dfef6dcp-5, 0x1.baf2619083409p-7},
    {0x1.c72fb46bcb00dp-7, -0x1.563eff5bfb105p-12}, {0x1.0b5fa5ab5e321p-4, 0x1.4ab38d0e12162p-4},
    {-0x1.380d16b6035ecp-9, -0x1.58dd9357e84f7p-9}, {-0x1.1551164752c4ap-7, -0x1.38f74e48f8e33p-6},
    {-0x1.9f40b3ab9915cp-6, 0x1.9fac67b425a0bp-7}, {0x1.331ec674936fap-10, 0x1.c248ab8b4b00fp-7},
    {0x1.57f19db5975dep-8, 0x1.d578b91572322p-6}, {0x1.d49fcfaa0df4ap-7, 0x1.3ccbc39b7d00ap-9},
    {0x1.8cd5464abc427p-4, 0x1.f36f82a3c2016p-4}, {-0x1.3e966cd1305b5p-9, -0x1.240f6e9ea5293p-7},
    {-0x1.24afc686c9529p-6, 0x1.e874265ec8b82p-5}, {0x1.237c9c6a14ab4p-6, -0x1.0d0b4be621a69p-4},
    {-0x1.32ba122db91f5p-6, -0x1.98602f1dfa910p-6}, {-0x1.700dfcc09782ap-10, -0x1.2e3f4e05cee83p-7},
    {0x1.b02a42579963bp-6, -0x1.3033dd1602943p-5}, {0x1.5841851a3a3cdp-8, 0x1.4fc55df6dfca0p-7},
    {-0x1.6383514c091dfp-5, 0x1.26510e13e40c6p-4}, {-0x1.8d5db1be0f828p-10, -0x1.69bb5ffde6483p-5},
    {0x1.96e02f7611d7fp-7, 0x1.01b908ceceb70p-7}, {-0x1.1d78998e01d75p-4, 0x1.c507f4bfcaf65p-4},
    {-0x1.81ec8a23719c4p-7, 0x1.7e7ae21341976p-6}, {-0x1.36aeb5dc93700p-6, -0x1.3ab848a07f66cp-7},
    {0x1.6534138ca045dp-4, 0x1.40b73921f4bf4p-5}, {0x1.af43a94465823p-8, 0x1.136fde1687301p-6},
    {0x1.2df197a423a84p-9, 0x1.2ffcce75c1ac4p-7}, {0x1.6ba14db7844e5p-7, 0x1.acf6010dc4750p-9},
    {0x1.c636a1b5a0210p-6, -0x1.f0008ad9417e7p-6}, {0x1.be43e693db78bp-7, -0x1.2bea533ea84c6p-9},
    {0x1.5e91ae1309e6dp-8, -0x1.8ecd2cf952726p-8}, {-0x1.e6d43a39ffcfep-7, -0x1.1ad81e3093925p-5},
    {-0x1.d125e5bc1dae9p-9, -0x1.ceb97f6992437p-7}, {-0x1.c93fa9b35e532p-6, -0x1.4dce5db098178p-7},
    {0x1.fef19ed0c2b38p-4, 0x1.09782fd242701p-3}, {0x1.d9e47a07b5c4fp-9, 0x1.78d6f7a6e0c4ap-7},
    {-0x1.eb69edfcc6850p-7, 0x1.4a38a354bf9fap-8}, {-0x1.e70bf6716d82cp-7, -0x1.2cd04b9574af1p-4},
    {0x1.0843d5102e32fp-5, -0x1.d3504bea9bb9dp-7}, {0x1.d89b50c04cb5bp-10, -0x1.c059b0f672791p-7},
    {0x1.e5e9c52830fa2p-4, -0x1.e9647b92221cdp-4}, {-0x1.80ff6006b2514p-8, -0x1.be00f814af635p-8},
    {0x1.6303f2d621007p-7, 0x1.b431cd5fabdb1p-7}, {-0x1.44d31fef82556p-8, -0x1.dbd927bf08800p-6},
    {0x1.47a8346434b21p-4, 0x1.d8683b939165ep-9}, {-0x1.2360d1be81ed9p-8, -0x1.7feccb7e01ac4p-11},
    {0x1.0fe5a249b0dc5p-7, 0x1.1b3673045d4acp-6}, {-0x1.cf80237fbde8ap-6, -0x1.9723bc38ae86ep-6},
    {0x1.4c60450273b42p-6, -0x1.bd7ae85cdc3b9p-8}, {-0x1.070f4512c7014p-7, 0x1.7022c67127e1bp-14},
    {-0x1.44a376814c6e5p-3, 0x1.82daac0d4bebfp-5}, {-0x1.67b009fffa1bbp-6, -0x1.31743b3bde3c9p-6},
    {-0x1.1837a6436600bp-7, 0x1.9003a45348cc6p-8}, {0x1.db800148c4dd8p-6, 0x1.1c4b78543b2ecp-4},
    {0x1.f2a2d32477c9dp-5, -0x1.3f1636d085636p-8}, {-0x1.5e24a0243f9d6p-7, 0x1.a2364409ce712p-7},
    {-0x1.484a273c65b0cp-4, -0x1.580d330bd388bp-4}, {-0x1.1644abcc72698p-5, -0x1.6aaa69d773c40p-6},
    {0x1.e2ec1f66d8922p-6, 0x1.26a8af91e1b9fp-7}, {0x1.c8bff17d4b80bp-5, 0x1.bdac30a37f6adp-7},
    {-0x1.49cc3cb76314ap-4, -0x1.02c4617f0a3a7p-5}, {0x1.39ed59439e382p-8, -0x1.6ed7f235d0cfep-6},
    {0x1.4b2a4813faa18p-6, 0x1.58f844b4b2888p-5}, {-0x1.c18c31cfd9ac2p-5, 0x1.b9918e84d9366p-8},
    {0x1.9faaab8569e46p-7, -0x1.0d192ad6cd4e1p-6}, {-0x1.fd0fe42de2e2dp-5, -0x1.208277c1edfbcp-4},
    {0x1.357f70d4f8380p-8, 0x1.ad0e2a1ec3a9dp-5}, {0x1.bbb02a5713034p-6, 0x1.29ee405714e4ap-7},
    {0x1.edbfa9d17c79bp-7, 0x1.cabcf20ee4786p-6}, {0x1.ea804efab7a71p-4, 0x1.c32624802fb3dp-5},
    {0x1.6f773f61626e8p-6, 0x1.1daf629f0f1d2p-7}, {-0x1.5ad5753d2d62dp-10, 0x1.2bc1be45649e3p-3},
    {-0x1.ca1c2a3d7369bp-9, 0x1.f1c8a4eceb40fp-9}, {-0x1.18d04e8ef1cb2p-8, -0x1.75c372fe3896ap-6},
    {0x1.3862198f80bdep-6, 0x1.ae10c30cd0a3ep-4}, {-0x1.6a8535d6c9991p-7, -0x1.c19387a3a3b2ap-8},
    {0x1.ca1bbf0210b8dp-8, -0x1.102cde18fc4e2p-7}, {-0x1.8a8f189ff46d6p-8, 0x1.ce688c497d4ddp-5},
    {0x1.3d2536245bdecp-13, 0x1.8e8ade9dbb6cfp-7}, {0x1.3e4f746219fb1p-10, -0x1.232cbd5b9eb41p-13},
    {0x1.6c4c96cdcb6e0p-9, 0x1.79ff7ef394265p-7}, {-0x1.b4650f03c71c7p-4, -0x1.b1b8b30cf53ffp-4},
    {-0x1.552fa2dfef6cap-5, 0x1.baf26190833bep-7}, {-0x1.d9b601703ce76p-8, -0x1.84daf7de8a26dp-7},
    {0x1.316aff1c40728p-5, -0x1.8ce72d3fde02ep-4}, {-0x1.1d460383dbcb3p-10, 0x1.baad4dd254d7fp-9},
    {0x1.545da03be0c18p-6, 0x1.2335f4ad9d76dp-9}, {0x1.ba26d89303af5p-10, -0x1.cf89ac716e739p-6},
    {-0x1.9926fbeef0134p-7, -0x1.7fca5705b5560p-8}, {0x1.57f19db5975a6p-8, 0x1.d578b91572352p-6},
    {0x1.d49fcfaa0df04p-7, 0x1.3ccbc39b7cd6ap-9}, {0x1.8cd5464abc408p-4, 0x1.f36f82a3c2029p-4},
    {-0x1.aa3762888e349p-8, 0x1.ae03441fb0b07p-8}, {0x1.f02f5ab146f57p-5, -0x1.d5f595f152ccbp-7},
    {-0x1.0d6f5533a57f6p-4, 0x1.1da73f89a6727p-6}, {0x1.fb06f467f64efp-6, -0x1.eb90213b9d44cp-9},
    {0x1.1cc1da909780ap-7, 0x1.bd1f4ad0c1efdp-9}, {0x1.36cfd2a926a1dp-6, 0x1.533bf8a61b036p-5},
    {0x1.5841851a3a3a0p-8, 0x1.4fc55df6dfc78p-7}, {-0x1.6383514c091cfp-5, 0x1.26510e13e40d9p-4},
    {-0x1.8d5db1be0f5c0p-10, -0x1.69bb5ffde6483p-5}, {0x1.3c1b1f643f035p-11, -0x1.e139e241e6f62p-7},
    {0x1.0b89364843ed4p-3, 0x1.4b5ad8dcb39b1p-8}, {0x1.abb7e93c8179fp-6, -0x1.82148f5e2d5a4p-10},
    {0x1.239e7206608f5p-6, -0x1.7cc2047c68b87p-7}, {-0x1.3d79c6d4b3881p-4, 0x1.ca562cbaef307p-5},
    {-0x1.247186b510a8ep-6, -0x1.62c6b836bb982p-9}, {0x1.2df197a423a84p-9, 0x1.2ffcce75c1aefp-7},
    {0x1.6ba14db7844bfp-7, 0x1.acf6010dc47ecp-9}, {0x1.c636a1b5a020cp-6, -0x1.f0008ad9417dap-6},
    {-0x1.2010f2034143ap-7, -0x1.5cfcd45f6fd89p-7}, {-0x1.04540fd4c1f91p-7, -0x1.a0cdcdb90a6c9p-10},
    {-0x1.70317b761e79dp-6, 0x1.eda5b9329068cp-6}, {0x1.cadff0803303cp-7, 0x1.054f416ffdca6p-8},
    {0x1.752aa847d8be5p-6, -0x1.388996130ff81p-6}, {-0x1.65a3a73f4ec70p-3, 0x1.620aadb8588eep-5},
    {0x1.d9e47a07b5c48p-9, 0x1.78d6f7a6e0c30p-7}, {-0x1.eb69edfcc689dp-7, 0x1.4a38a354bf9bep-8},
    {-0x1.e70bf6716d794p-7, -0x1.2cd04b9574aeep-4}, {-0x1.d29e22e0a3980p-6, -0x1.54e4562b01aa7p-6},
    {-0x1.a1d21a87c121cp-7, 0x1.5a072b49286cbp-8}, {-0x1.4d6437035536dp-3, -0x1.603bd8eb104edp-5},
    {0x1.215febf0e525ep-7, -0x1.b9a9c099d30c6p-10}, {-0x1.15a1b2a86aa70p-6, 0x1.656ba0ead7549p-9},
    {0x1.c4b324441ce78p-6, 0x1.4f31ee157a0a6p-7},
};
// clang-format on
constexpr int kBlockRows = 9;
constexpr int kBlockCols = 27;

std::uint64_t mix(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Moves later allocations: keeps a block of pseudo-random size alive and
// sometimes frees the oldest, so two consecutive calls see different
// distances between the buffers they allocate.
class HeapShuffle {
public:
    void step() {
        held_.emplace_back(new char[1 + rng_() % 4096]);
        if (rng_() % 3 == 0) held_.erase(held_.begin());
    }

private:
    std::mt19937_64 rng_{12345};
    std::vector<std::unique_ptr<char[]>> held_;
};

std::uint64_t factorise(const std::vector<Complex128>& block) {
    const std::vector<Complex128> copy = block;  // at a fresh address each call
    const detail::SvdTruncation r = detail::svd_truncate_verified(
        copy.data(), kBlockRows, kBlockCols, detail::MatrixOrder::ColMajor, 64,
        MPS_DEFAULT_CUTOFF, SVDMethod::BDC,
        detail::SvdPolicy{SvdRejection::Fix, false, SvdReport::Silent}, "V11313");
    std::uint64_t h = 1469598103934665603ULL;
    h = mix(h, r.U.data(), sizeof(Complex128) * static_cast<std::size_t>(r.U.size()));
    h = mix(h, r.S.data(), sizeof(double) * static_cast<std::size_t>(r.S.size()));
    h = mix(h, r.V.data(), sizeof(Complex128) * static_cast<std::size_t>(r.V.size()));
    h = mix(h, &r.rank, sizeof r.rank);
    const bool flags[3] = {r.used_jacobi_rescue, r.used_gram_fallback, r.used_unverified};
    return mix(h, flags, sizeof flags);
}

// The chain from test_v11311_svd_settings.cpp: seven qutrits, 168 gates,
// every pair distance. Hashes every tensor and the rescue count.
std::uint64_t qutrit_chain() {
    constexpr int n = 7, d = 3;
    const auto F = qudit_gates::qft_matrix(d);
    const auto SUM = qudit_gates::cadd_matrix(d, 1);
    std::vector<Complex128> CP(81, Complex128(0.0, 0.0));
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            const std::size_t k = static_cast<std::size_t>(a + d * b);
            const double phi = TWO_PI * a * b / d;
            CP[k * 9 + k] = Complex128(std::cos(phi), std::sin(phi));
        }
    QuditMPS mps(n, d, 64);
    mps.svd_report = SvdReport::Silent;
    for (int step = 0; step < 56; ++step) {
        mps.apply_1qudit((3 * step + 1) % n, F);
        const int a = (5 * step) % n;
        int b = (2 * step + 3) % n;
        if (b == a) b = (b + 1) % n;
        mps.apply_2qudit(a, b, (step % 2 == 0) ? SUM : CP);
        const int c0 = (step * 4 + 2) % n;
        int c1 = (n - 1) - (step % n);
        if (c1 == c0) c1 = (c1 + 1) % n;
        mps.apply_2qudit(c0, c1, (step % 3 == 0) ? CP : SUM);
    }
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto& t : mps.tensors()) {
        h = mix(h, t.data.data(), t.data.size() * sizeof(Complex128));
    }
    const std::size_t rescues = mps.jacobi_rescue_count();
    return mix(h, &rescues, sizeof rescues);
}

}  // namespace

TEST(V11313Reproducibility, ADegenerateBlockFactorisesTheSameAtEveryHeapLayout) {
    ASSERT_EQ(kDegenerateBlock.size(), static_cast<std::size_t>(kBlockRows * kBlockCols));
    HeapShuffle heap;
    std::set<std::uint64_t> seen;
    for (int call = 0; call < 300; ++call) {
        heap.step();
        seen.insert(factorise(kDegenerateBlock));
    }
    EXPECT_EQ(seen.size(), 1u) << "the factorisation depends on where the heap put its buffers";
}

TEST(V11313Reproducibility, AQutritChainIsTheSameChainEveryTimeInOneProcess) {
    HeapShuffle heap;
    std::set<std::uint64_t> seen;
    for (int run = 0; run < 30; ++run) {
        heap.step();
        seen.insert(qutrit_chain());
    }
    EXPECT_EQ(seen.size(), 1u) << "the chain depends on where the heap put its buffers";
}
