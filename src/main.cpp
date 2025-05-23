/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "AllGather.h"
#include "Hypercube3D.h"
#include "LinkUsageTracker.h"
#include "Mesh2D.h"
#include "TacosGreedy.h"
#include "Timer.h"
#include "Torus2D.h"
#include "Torus3D.h"
#include <iostream>

using namespace Tacos;

int main() {
    // set print precision
    fixed(std::cout);
    std::cout.precision(2);

    // construct a topology
    //    const auto x = 8;
    const auto x = 5;
    const auto y = x;
    const auto z = y;

    const auto bw = 50;
    const auto bw_beta = 1'000'000 / (bw * 1024.0);
    std::cout << bw_beta << std::endl;

    //    const auto linkAlphaBeta = std::make_pair(0, 1);
    const auto linkAlphaBeta = std::make_pair(0.5, bw_beta);

    //    const auto topology = std::make_shared<Mesh2D>(x, y, linkAlphaBeta);
    //    const auto filename = "../../Mesh2D_" + std::to_string(x) + "_" + std::to_string(y) + ".csv";

    //    const auto topology = std::make_shared<Torus2D>(x, y, linkAlphaBeta);
    //    const auto filename = "../../Torus2D_" + std::to_string(x) + "_" + std::to_string(y) + ".csv";

    const auto topology = std::make_shared<Hypercube3D>(x, y, z, linkAlphaBeta);
    const auto filename =
        "../../Hypercube3D_" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z) + ".csv";

    //    const auto topology = std::make_shared<Torus3D>(x, y, z, linkAlphaBeta);
    //    const auto filename = "../../Torus3D_" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z)
    //    + ".csv"; const auto filename = "../../Torus3D_" + std::to_string(x) + "_" + std::to_string(y) + "_" +
    //    std::to_string(z) + ".csv";

    const auto npusCount = topology->getNpusCount();
    std::cout << "NPUs count: " << npusCount << std::endl;

    // create collective
    const auto collectivesCount = 1;
    const auto chunkSize = 1024.0 / (npusCount * collectivesCount);
    //    const auto chunkSize = 0.25;
    //    const auto chunkSize = 1;
    //    const auto chunkSize = 1024 / npusCount;
    //    const auto chunkSize = 8;
    //    const auto collectivesCount = 2;
    const auto collective = std::make_shared<AllGather>(npusCount, chunkSize, collectivesCount);
    const auto chunksCount = collective->getChunksCount();
    std::cout << "Chunks count: " << chunksCount << std::endl;

    // create collective algorithm stat monitor
    auto linkUsageTracker = std::make_shared<LinkUsageTracker>();

    // create timer
    auto solverTimer = Timer("PathSolver");

    // create solver and solve
    solverTimer.start();
    auto solver = TacosGreedy(topology, collective, linkUsageTracker);
    auto collectiveTime = solver.solve();
    solverTimer.stop();

    // print result
    auto time = solverTimer.getTime("ms");
    std::cout << std::endl;
    std::cout << "Time to solve: " << time << " ms" << std::endl;
    std::cout << "All-Gather Time: " << collectiveTime << std::endl;
    std::cout << "All-Reduce Time: " << collectiveTime * 2 << std::endl;

    const auto linksCount = topology->getLinksCount();
    std::cout << "Links: " << linksCount << std::endl;

    // save link usage
    linkUsageTracker->saveLinkUsage(filename, linksCount, collectiveTime);

    // terminate
    return 0;
}
