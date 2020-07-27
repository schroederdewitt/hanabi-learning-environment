// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The standard Open Hanabi observation encoders. These encoders translate
// HanabiObservations to input tensors that an agent can train on.

#ifndef __CANONICAL_ENCODERS_H__
#define __CANONICAL_ENCODERS_H__

#include <vector>

#include "hanabi_game.h"
#include "hanabi_observation.h"
#include "observation_encoder.h"

namespace hanabi_learning_env {

// This is the canonical observation encoding.
class CanonicalObservationEncoder : public ObservationEncoder {
 public:
  explicit CanonicalObservationEncoder(const HanabiGame* parent_game)
      : parent_game_(parent_game) {}

  std::vector<int> Shape() const override;

  std::vector<int> ShapeJointObs() const;

  // std::vector<float> Encode(const HanabiObservation&) const override {
  //   // std::cerr << "Not Impled" << std::endl;
  //   // throw std::logic_error("Function not yet implemented"); //assert(false);
  //   // return std::vector<float>();
  //   // // return Encode(obs, false, false);
  //   return Encode(obs, false, {}, false, {}, {}, false);
  // }

  std::vector<float> Encode(const HanabiObservation& obs,
                            bool show_own_cards,
                            const std::vector<int>& order,
                            bool shuffle_color,
                            const std::vector<int>& color_permute,
                            const std::vector<int>& inv_color_permute,
                            bool hide_action) const;

  // std::vector<float> EncodeV0Belief(const HanabiObservation& obs, bool all_player) const;
  // std::vector<float> EncodeV1Belief(const HanabiObservation& obs, bool all_player) const;
  // std::vector<float> EncodeHandMask(const HanabiObservation& obs) const;
  // std::vector<float> EncodeCardCount(const HanabiObservation& obs) const;

  std::vector<float> EncodeLastAction(
      const HanabiObservation& obs,
      const std::vector<int>& order,
      bool shuffle_color,
      const std::vector<int>& color_permute) const;

  // for aux task
  std::vector<float> EncodeOwnHandTrinary(const HanabiObservation& obs) const;

  std::vector<float> EncodeOwnHand(
      const HanabiObservation& obs,
      bool shuffle_color,
      const std::vector<int>& color_permute) const;

  std::vector<float> EncodeAllHand(
      const HanabiObservation& obs,
      bool shuffle_color,
      const std::vector<int>& color_permute) const;

  std::vector<float> EncodeJointFivePlayers(const HanabiObservation& obs,
                                            bool show_own_cards,
                                            const std::vector<int>& order,
                                            bool shuffle_color,
                                            const std::vector<int>& color_permute,
                                            const std::vector<int>& inv_color_permute,
                                            bool hide_action) const;

  // std::vector<std::vector<int>> ComputePrivateCardCount(
  //   const HanabiObservation& obs,
  //   bool shuffle_color,
  //   const std::vector<int>& color_permute) const;

  ObservationEncoder::Type type() const override {
    return ObservationEncoder::Type::kCanonical;
  }

 private:
  const HanabiGame* parent_game_ = nullptr;
};

int LastActionSectionLength(const HanabiGame& game,
                            bool using_joint_obs = false);

}  // namespace hanabi_learning_env

#endif
