GENERATIONS := 20
GENERATION_SIZE := 100
OUTPUT := artifacts/checkpoints/run_acceleration.json
PLOT := artifacts/plots/loss_acceleration.png
MODE := single
CHECKPOINT := $(OUTPUT)
EXPERIMENT_NAME := incline

START_POSITIONS := front middle rear
MUTATIONS := 0.01 0.04 0.1
CHAMPIONS_LIST := 6 20 50
# 27 experiments

.PHONY: train playback load experiment run_single_exp

load:
	pip install -e '.[dev,notebook,viz]'

# The "Outer Loop"
experiment:
	@$(foreach sp,$(START_POSITIONS),\
		$(foreach mut,$(MUTATIONS),\
			$(foreach champ,$(CHAMPIONS_LIST),\
				$(MAKE) run_single_exp START_POSITION=$(sp) MUTATION=$(mut) CHAMPIONS=$(champ); \
			)))

# The "Inner Execution" where variables are applied
run_single_exp:
	@echo "Running: SP=$(START_POSITION), MUT=$(MUTATION), CHAMP=$(CHAMPIONS)"
	@$(MAKE) train START_POSITION=$(START_POSITION) MUTATION=$(MUTATION) CHAMPIONS=$(CHAMPIONS)
	@$(MAKE) playback START_POSITION=$(START_POSITION) MUTATION=$(MUTATION) CHAMPIONS=$(CHAMPIONS)

train:
	spce train --generations $(GENERATIONS) \
	   --generation-size $(GENERATION_SIZE) \
	   --output $(OUTPUT) \
	   --plot $(PLOT) \
	   --mode $(MODE) \
	   --start-position $(START_POSITION) \
	   --mutation $(MUTATION) \
	   --champions $(CHAMPIONS)

playback:
	$(eval CURRENT_GIF := artifacts/playback/$(EXPERIMENT_NAME)_g$(GENERATIONS)_sp$(START_POSITION)_m$(MUTATION)_c$(CHAMPIONS).gif)
	spce playback --checkpoint $(CHECKPOINT) \
	   --output $(CURRENT_GIF)