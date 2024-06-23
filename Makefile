SRC := bee_rl/*.py bee_rl/*/*.py
TEST_SRC := tests/unit/*.py tests/integration/*.py

fmt:
	@black $(SRC) $(TEST_SRC)

.PHONY: fmt
