/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "Topology.h"

namespace Tacos {
class Mesh2DHetero final : public Topology {
  public:
    Mesh2DHetero(int width,
                 int height,
                 LinkAlphaBeta linkAlphaBeta1,
                 LinkAlphaBeta linkAlphaBeta2) noexcept;
};
}  // namespace Tacos
