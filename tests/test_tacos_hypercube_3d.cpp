/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <AllGather.h>
#include <Hypercube3D.h>
#include <TacosGreedy.h>
#include <gtest/gtest.h>
#include <test_utils.h>

using namespace Tacos;

TEST(Hypercube3DTest, Hypercube3x3x3) {
    const auto x = 3;
    const auto y = 3;
    const auto z = 3;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Hypercube3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 1;
    const auto chunkSize = 1024.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto samples = std::vector<Tacos::Time>();
    for (int i = 0; i < 10; ++i) {
        auto solver = TacosGreedy(topology, collective);
        auto collectiveTime = solver.solve();
        samples.push_back(collectiveTime);
    }

    const auto expected = 6671.17;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Hypercube3DTest, Hypercube3x4x5) {
    const auto x = 3;
    const auto y = 4;
    const auto z = 5;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Hypercube3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 2;
    const auto chunkSize = 2048.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto samples = std::vector<Tacos::Time>();
    for (int i = 0; i < 10; ++i) {
        auto solver = TacosGreedy(topology, collective);
        auto collectiveTime = solver.solve();
        samples.push_back(collectiveTime);
    }

    const auto expected = 13353.33;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Hypercube3DTest, Hypercube4x2x3) {
    const auto x = 4;
    const auto y = 2;
    const auto z = 3;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Hypercube3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 3;
    const auto chunkSize = 3072.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto samples = std::vector<Tacos::Time>();
    for (int i = 0; i < 10; ++i) {
        auto solver = TacosGreedy(topology, collective);
        auto collectiveTime = solver.solve();
        samples.push_back(collectiveTime);
    }

    const auto expected = 20012.0;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}
