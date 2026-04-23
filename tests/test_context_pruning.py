""" Tests for context pruning strategy."""


class TestPruneContext:

    def test_no_pruning_needed(self, pruning):
        ctx = ["Block 1", "Block 2", "Block 3"]
        result = pruning.prune(ctx, "Test question", [], max_blocks=5, n_new=1)
        assert result == ctx

    def test_pruning_keeps_max_blocks(self, pruning):
        ctx = [f"Block {i}" for i in range(10)]
        result = pruning.prune(ctx, "question about topic", [], max_blocks=5, n_new=2)
        assert len(result) == 5

    def test_new_blocks_protected(self, pruning):
        ctx = ["Old irrelevant block", "Old relevant question block", "New block about question"]
        result = pruning.prune(ctx, "question", [], max_blocks=2, n_new=1)
        assert len(result) == 2
        assert "New block about question" in result

    def test_relevance_based_scoring(self, pruning):
        ctx = [
            "This block is about cats and dogs",
            "This block discusses the main question topic",
            "Another irrelevant block about weather",
            "New block just added",
        ]
        result = pruning.prune(ctx, "question topic", [], max_blocks=2, n_new=1)
        assert len(result) == 2
        assert "New block just added" in result
        assert "main question topic" in result[0]

    def test_empty_context(self, pruning):
        assert pruning.prune([], "question", [], max_blocks=5, n_new=0) == []

    def test_sub_questions_improve_scoring(self, pruning):
        ctx = [
            "Block about machine learning models",
            "Block about climate change impacts",
            "New block",
        ]
        result = pruning.prune(
            ctx, "What is AI?",
            sub_questions=["How do machine learning models work?"],
            max_blocks=2, n_new=1,
        )
        assert len(result) == 2
        assert "machine learning" in result[0]

    def test_punctuation_is_normalized_for_overlap_scoring(self, pruning):
        ctx = [
            "Alter irrelevanter Block ohne Bezug",
            "Relevanter Block mit Reform,",
            "Neuester Block",
        ]

        result = pruning.prune(ctx, "Reform?", [], max_blocks=2, n_new=1)

        assert len(result) == 2
        assert "Relevanter Block mit Reform," in result
        assert "Alter irrelevanter Block ohne Bezug" not in result

    def test_equal_relevance_prefers_more_recent_old_block(self, pruning):
        ctx = [
            "Alte Quelle: Frage zur Reform.",
            "Neuere Quelle: Frage zur Reform.",
            "Neuester Block",
        ]

        result = pruning.prune(ctx, "Frage zur Reform?", [], max_blocks=2, n_new=1)

        assert len(result) == 2
        assert "Neuere Quelle: Frage zur Reform." in result
        assert "Alte Quelle: Frage zur Reform." not in result
        assert result[-1] == "Neuester Block"

    def test_required_aspects_preserve_diverse_old_blocks(self, pruning):
        ctx = [
            "Patientenverbaende und Krankenkassen vertreten gegensaetzliche Positionen zur Reform.",
            "Kritiker sehen erhebliche Risiken, Probleme und Unsicherheiten bei der Umsetzung.",
            "Alte irrelevante Wetterquelle ohne Sachbezug.",
            "Neuester Block zum aktuellen Stand der Reform.",
        ]

        result = pruning.prune(
            ctx,
            "Frage zur Reform",
            [],
            max_blocks=3,
            n_new=1,
            required_aspects=[
                "Stakeholder- und Betroffenenperspektiven",
                "Gegenargumente, Risiken und Limitationen",
            ],
        )

        assert len(result) == 3
        assert ctx[0] in result
        assert ctx[1] in result
        assert ctx[3] in result
