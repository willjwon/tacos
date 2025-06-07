/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <AllGather.h>
#include <Mesh2D.h>
#include <TacosGreedy.h>
#include <gtest/gtest.h>
#include <iostream>
#include <test_utils.h>

using namespace Tacos;

TEST(Mesh2DTest, Mesh5x5) {
    const auto width = 5;
    const auto height = 5;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Mesh2D>(width, height, linkAlphaBeta);
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

    const auto expected = 9606.0;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Mesh2DTest, Mesh10x10) {
    const auto width = 10;
    const auto height = 10;
    const auto alpha = 1;
    const auto bandwidth = 100.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Mesh2D>(width, height, linkAlphaBeta);
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

    const auto expected = 5049.0;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}

TEST(Mesh2DTest, Mesh7x12) {
    const auto width = 7;
    const auto height = 12;
    const auto alpha = 1;
    const auto bandwidth = 100.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Mesh2D>(width, height, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, width * height);

    const auto collectivesCount = 3;
    const auto chunkSize = 4096.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto solver = TacosGreedy(topology, collective);
    auto collectiveTime = solver.solve();

    auto samples = std::vector<Tacos::Time>();
    for (int i = 0; i < 10; ++i) {
        auto solver = TacosGreedy(topology, collective);
        auto collectiveTime = solver.solve();
        samples.push_back(collectiveTime);
    }

    const auto expected = 19966.27;
    const auto counts = count_within_tolerance(samples, expected, 0.05);
    ASSERT_GE(counts, 7);
}
