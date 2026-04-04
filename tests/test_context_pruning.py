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
