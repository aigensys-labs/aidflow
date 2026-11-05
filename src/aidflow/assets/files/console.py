"""
AIDflow — End-to-End Console Demo.

A self-contained script that demonstrates the full AIDflow
lifecycle using synthetic data:

    0. Generate demo data
    1. Define project context (genesis + params)
    2. Run the pipeline (by phase or by step)
    3. Predict with the champion model
    4. Export the production package

Run directly:
    python console.py

Or import individual steps for interactive use:
    from console import gen_data, set_context, run_flow, predict, export
"""

from aidflow import (
    ProjectContext,
    Predictor,
    ProductionExporter
)    
from aidflow.assets import (
    DataGetter
)


#════════════════════════════════════════════════════════════════════
# 0 DEMO DATA
#════════════════════════════════════════════════════════════════════
def create_data(size: str = 'small') -> str:
	"""
    Generate synthetic demo data and save to disk.

    Parameters
    ----------
    size : 'small' | 'medium' | 'large', default='small'
        Dataset scale.

    Returns
    -------
    str
        File path to the generated CSV.
    """    	
	return DataGetter.create_data(size, 'data/test.csv')
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 1 PROJECT DEFINITION
#════════════════════════════════════════════════════════════════════
def set_context(data_path: str) -> ProjectContext:
	"""
    Initialize the project context.

    Defines the project identity (genesis) and flow
    parameters (save, logging, verbosity).

    Parameters
    ----------
    data_path : str
        Path to the input CSV file.

    Returns
    -------
    ProjectContext
        Configured context, ready for execution.
    """
	ctx = ProjectContext()

	ctx.genesis(
		project		 = 'CONSOLE', 
		input_data 	 = data_path, 
		target 		 = 'target', 
		yy_mm 		 = 'time', 
		yy_mm_format = '%Y-%m-%d'
	)
	ctx.params(		
		save	= 1,
		logging	= 1, 
		verbose	= 1, 
	)

	return ctx
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 2-A RUN BY PHASES
#════════════════════════════════════════════════════════════════════
def run_flow(ctx: ProjectContext, phase = 'all') -> None:
	"""
    Execute the pipeline by phase (sequence-dependent).

    Intended for automated re-training loops in CLI.

    Parameters
    ----------
    ctx : ProjectContext
        Configured project context.
    phase : str, default='all'
        Phase to execute: 'all' | 'data' | 'encoding' |
        'training' | 'selection'.
    """
	ctx.flow.process(phase)
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 2-B RUN BY STEPS
#════════════════════════════════════════════════════════════════════
def run_steps(ctx: ProjectContext) -> None:
	"""
    Execute the pipeline step-by-step, retaining each worker.

    Intended for interactive exploration in notebooks — each
    worker instance exposes `.inputs`, `.result`, `.stats`,
    and `.report` for inspection and fine-tuning.
	"""

	# ── DATA ───────────────────────────────────────────────────────
	data = ctx.flow.data

	data_loader = data.data_loader()
	fe_cleaner = data.fe_cleaner()
	fe_stabilizer = data.fe_stabilizer()
	obs_stabilizer = data.obs_stabilizer()


    # ── ENCODING ───────────────────────────────────────────────────
	encoding = ctx.flow.encoding

	num_encoder = encoding.num_encoder()
	cat_encoder = encoding.cat_encoder()
	ixn_encoder = encoding.ixn_encoder()
	woe_encoder = encoding.woe_encoder()
	woe_stabilizer = encoding.woe_stabilizer()
	data_encoder = encoding.data_encoder()


    # ── TRAINING ───────────────────────────────────────────────────
	training = ctx.flow.training

	agent1_predictor = training.agent1_predictor()
	agent2_predictor = training.agent2_predictor()
	agent3_predictor = training.agent3_predictor()
	agent4_predictor = training.agent4_predictor()


    # ── SELECTION ──────────────────────────────────────────────────
	selection = ctx.flow.selection

	agents_predictor = selection.agents_predictor()
	agents_comparator = selection.agents_comparator()
	agents_transition = selection.agents_transition()
	agent_selector = selection.agent_selector()
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 3 PREDICTION
#════════════════════════════════════════════════════════════════════
def predict(ctx: ProjectContext) -> None:
	"""
    Score new data with the trained champion model.

    The `Predictor` applies the full preprocessing pipeline
    (clean → encode → predict) using the parameters locked
    during training.

	Only applicable for `persited` trained models (ctx save=1).
    """
	setup = ctx.setup	
	pred  = Predictor(setup.project) # or use ctx.predictor
	data  = ctx.data_io.load(setup.input_data) # or available scoring data

	p_hat = pred.champion_predict(data, preprocess=True).data.squeeze()
	print(
		'► Champion Prediction → '
		f'E(p): {p_hat.mean():.3f} | S(p): {p_hat.std():.3f}'
	)
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 4 PRODUCTION EXPORT
#════════════════════════════════════════════════════════════════════
def export(ctx: ProjectContext) -> None:
	"""
    Export the champion model as a standalone production artifact.

    The exported package is fully self-contained — it includes
    all encoding maps, feature transformations, and model
    parameters needed for inference without the AIDflow
    framework.
    """
	pred = ctx.workers.champion_predictor
	prod = ProductionExporter(pred)

	path = f'{ctx.structure.root}/production'
	exp  = prod.export_all(path)
	print(f'► Production Export → {exp}')
#════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════
# 5 ENTRY POINT
#════════════════════════════════════════════════════════════════════
def main(phase = 'all', size: str = 'small') -> None:
	"""
    Execute the full AIDflow demo lifecycle.

    Generates synthetic data, initializes the project context,
    runs the training pipeline, and (if phase='all') scores
    with the champion model and exports the production artifact.

    Parameters
    ----------
    phase : str, default='all'
        Pipeline scope. `'all'` runs training + prediction +
        export. A specific phase name (e.g., `'data'`) runs
        only that stage.
    size : str, default='small'
        Synthetic data scale: 'small' | 'medium' | 'large'.
    """	
	path = create_data(size)
	ctx = set_context(path)
	run_flow(ctx)
	
	if phase == 'all':
		predict(ctx)
		export(ctx)	

if __name__ == "__main__":
	main()
#════════════════════════════════════════════════════════════════════