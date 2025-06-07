#include <AllGather.h>
#include <TacosGreedy.h>
#include <Torus3D.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace Tacos;

TEST(Torus2DTest, Torus3x3x3) {
    const auto x = 3;
    const auto y = 3;
    const auto z = 3;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Torus3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 1;
    const auto chunkSize = 1024.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto solver = TacosGreedy(topology, collective);
    auto collectiveTime = solver.solve();

    const auto expected = 3706.20;
    const auto tolerance = expected * 0.07;  // 5% tolerance
    ASSERT_NEAR(collectiveTime, expected, tolerance);
}

TEST(Torus2DTest, Torus3x4x5) {
    const auto x = 3;
    const auto y = 4;
    const auto z = 5;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Torus3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 2;
    const auto chunkSize = 2048.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto solver = TacosGreedy(topology, collective);
    auto collectiveTime = solver.solve();

    const auto expected = 7010.50;
    const auto tolerance = expected * 0.07;  // 5% tolerance
    ASSERT_NEAR(collectiveTime, expected, tolerance);
}

TEST(Torus2DTest, Torus4x2x3) {
    const auto x = 4;
    const auto y = 2;
    const auto z = 3;
    const auto alpha = 0.5;
    const auto bandwidth = 50.0;
    const auto beta = 1'000'000 / (bandwidth * 1024.0);
    const auto linkAlphaBeta = std::make_pair(alpha, beta);

    const auto topology = std::make_shared<Torus3D>(x, y, z, linkAlphaBeta);
    const auto npusCount = topology->getNpusCount();
    ASSERT_EQ(npusCount, x * y * z);

    const auto collectivesCount = 3;
    const auto chunkSize = 3072.0 / (npusCount * collectivesCount);
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    ASSERT_EQ(chunksCount, npusCount * collectivesCount);

    auto solver = TacosGreedy(topology, collective);
    auto collectiveTime = solver.solve();

    const auto expected = 12507.50;
    const auto tolerance = expected * 0.07;  // 5% tolerance
    ASSERT_NEAR(collectiveTime, expected, tolerance);
}
