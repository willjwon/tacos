/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <AllGather.h>
#include <Mesh2D_hetero.h>
#include <TacosGreedy.h>
#include <gtest/gtest.h>
#include <test_utils.h>

using namespace Tacos;

TEST(Mesh2DHeteroTest, Mesh2DHetero5x5) {
    const auto width = 5;
    const auto height = 5;
    const auto alpha = 0.5;
    const auto bandwidth1 = 50.0;
    const auto bandwidth2 = 75.0;
    const auto beta1 = 1'000'000 / (bandwidth1 * 1024.0);
    const auto beta2 = 1'000'000 / (bandwidth2 * 1024.0);
    const auto linkAlphaBeta1 = std::make_pair(alpha, beta1);
    const auto linkAlphaBeta2 = std::make_pair(alpha, beta2);

    const auto topology =
        std::make_shared<Mesh2DHetero>(width, height, linkAlphaBeta1, linkAlphaBeta2);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, width * height);

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

    const auto expected = 8005.0;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Mesh2DHeteroTest, Mesh2DHetero8x5) {
    const auto width = 8;
    const auto height = 5;
    const auto alpha = 1;
    const auto bandwidth1 = 80.0;
    const auto bandwidth2 = 35.0;
    const auto beta1 = 1'000'000 / (bandwidth1 * 1024.0);
    const auto beta2 = 1'000'000 / (bandwidth2 * 1024.0);
    const auto linkAlphaBeta1 = std::make_pair(alpha, beta1);
    const auto linkAlphaBeta2 = std::make_pair(alpha, beta2);

    const auto topology =
        std::make_shared<Mesh2DHetero>(width, height, linkAlphaBeta1, linkAlphaBeta2);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, width * height);

    const auto collectivesCount = 2;
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

    const auto expected = 8595.43;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Mesh2DHeteroTest, Mesh2DHetero7x10) {
    const auto width = 7;
    const auto height = 10;
    const auto alpha = 1.5;
    const auto bandwidth1 = 16.0;
    const auto bandwidth2 = 73.0;
    const auto beta1 = 1'000'000 / (bandwidth1 * 1024.0);
    const auto beta2 = 1'000'000 / (bandwidth2 * 1024.0);
    const auto linkAlphaBeta1 = std::make_pair(alpha, beta1);
    const auto linkAlphaBeta2 = std::make_pair(alpha, beta2);

    const auto topology =
        std::make_shared<Mesh2DHetero>(width, height, linkAlphaBeta1, linkAlphaBeta2);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, width * height);

    const auto collectivesCount = 3;
    const auto chunkSize = 4096.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto samples = std::vector<Tacos::Time>();
    for (int i = 0; i < 10; ++i) {
        auto solver = TacosGreedy(topology, collective);
        auto collectiveTime = solver.solve();
        samples.push_back(collectiveTime);
    }

    const auto expected = 44612.47;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}
