#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hanabi_card.h"
#include "hanabi_hand.h"
#include "hanabi_game.h"
#include "hanabi_move.h"
#include "hanabi_observation.h"
#include "canonical_encoders.h"

namespace py = pybind11;
using namespace hanabi_learning_env;

PYBIND11_MODULE(py_hanabi_lib, m) {
  py::class_<HanabiCard>(m, "HanabiCard")
      .def(py::init<int, int>())
      .def("color", &HanabiCard::Color)
      .def("rank", &HanabiCard::Rank)
      .def("is_valid", &HanabiCard::IsValid)
      .def("to_string", &HanabiCard::ToString)
      ;

  py::class_<HanabiHand::CardKnowledge>(m, "CardKnowledge")
      .def(py::init<int, int>())
      .def("num_colors", &HanabiHand::CardKnowledge::NumColors)
      .def("color_hinted", &HanabiHand::CardKnowledge::ColorHinted)
      .def("color", &HanabiHand::CardKnowledge::Color)
      .def("color_plausible", &HanabiHand::CardKnowledge::ColorPlausible)
      .def("apply_is_color_hint", &HanabiHand::CardKnowledge::ApplyIsColorHint)
      .def("apply_is_not_color_hint", &HanabiHand::CardKnowledge::ApplyIsNotColorHint)
      .def("num_ranks", &HanabiHand::CardKnowledge::NumRanks)
      .def("rank_hinted", &HanabiHand::CardKnowledge::RankHinted)
      .def("rank", &HanabiHand::CardKnowledge::Rank)
      .def("rank_plausible", &HanabiHand::CardKnowledge::RankPlausible)
      .def("apply_is_rank_hint", &HanabiHand::CardKnowledge::ApplyIsRankHint)
      .def("apply_is_not_rank_hint", &HanabiHand::CardKnowledge::ApplyIsNotRankHint)
      .def("is_card_plausible", &HanabiHand::CardKnowledge::IsCardPlausible)
      .def("to_string", &HanabiHand::CardKnowledge::ToString)
      ;

  py::class_<HanabiHand>(m, "HanabiHand")
      .def(py::init<>())
      .def("cards", &HanabiHand::Cards)
      .def("knowledge_", &HanabiHand::Knowledge_, py::return_value_policy::reference)
      .def("add_card", &HanabiHand::AddCard)
      .def("remove_from_hand", &HanabiHand::RemoveFromHand)
      ;

  py::class_<HanabiGame>(m, "HanabiGame")
      .def(py::init<const std::unordered_map<std::string, std::string>&>())
      .def("max_moves", &HanabiGame::MaxMoves)
      .def("get_move_uid",
           (int (HanabiGame::*)(HanabiMove) const) &HanabiGame::GetMoveUid)
      .def("get_move", &HanabiGame::GetMove)
      .def("num_colors", &HanabiGame::NumColors)
      .def("num_ranks", &HanabiGame::NumRanks)
      .def("hand_size", &HanabiGame::HandSize)
      .def("max_information_tokens", &HanabiGame::MaxInformationTokens)
      .def("max_life_tokens", &HanabiGame::MaxLifeTokens)
      .def("max_deck_size", &HanabiGame::MaxDeckSize)
      ;

  py::enum_<HanabiMove::Type>(m, "MoveType")
      .value("Invalid", HanabiMove::Type::kInvalid)
      .value("Play", HanabiMove::Type::kPlay)
      .value("Discard", HanabiMove::Type::kDiscard)
      .value("RevealColor", HanabiMove::Type::kRevealColor)
      .value("RevealRank", HanabiMove::Type::kRevealRank)
      .value("Deal", HanabiMove::Type::kDeal)
      ;
      // .export_values();

  py::class_<HanabiMove>(m, "HanabiMove")
      .def(py::init<HanabiMove::Type, int8_t, int8_t, int8_t, int8_t>())
      .def("move_type", &HanabiMove::MoveType)
      .def("target_offset", &HanabiMove::TargetOffset)
      .def("card_index", &HanabiMove::CardIndex)
      .def("color", &HanabiMove::Color)
      .def("rank", &HanabiMove::Rank)
      .def("to_string", &HanabiMove::ToString)
      ;

  py::class_<HanabiObservation>(m, "HanabiObservation")
      .def(py::init<
           int,
           int,
           const std::vector<HanabiHand>&,
           const std::vector<HanabiCard>&,
           const std::vector<int>&,
           int,
           int,
           int,
           const std::vector<HanabiMove>&,
           const HanabiGame*>())
      ;

  py::class_<CanonicalObservationEncoder>(m, "ObservationEncoder")
      .def(py::init<const HanabiGame*>())
      .def("shape", &CanonicalObservationEncoder::Shape)
      .def("encode", &CanonicalObservationEncoder::Encode)
      ;
}
