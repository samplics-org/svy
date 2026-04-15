# # tests/svy/regression/test_glm_core.py
# import logging

# import pytest

# import svy
# from svy.core.terms import Cat


# logging.basicConfig(
#     level=logging.DEBUG,  # <--- This reveals the "IRLS Iter..." lines
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )

# # 1. Setup Logging for the test file
# logger = logging.getLogger(__name__)


# @pytest.fixture(scope="module")
# def hld_sample():
#     """Load data and prepare the sample once for the module."""
#     # Load data
#     hld_data = svy.load_dataset(name="hld_sample_wb_2023", limit=None)
#     hld_design = svy.Design(stratum=("geo1", "urbrur"), psu="ea", wgt="hhweight")
#     sample = svy.Sample(data=hld_data, design=hld_design)

#     # Mutate data
#     sample = sample.wrangling.mutate(
#         {
#             "hhpovline": svy.col("hhsize") * 1800,
#             "pov_status": svy.when(svy.col("tot_exp") < svy.col("hhpovline")).then(1).otherwise(0),
#         }
#     )
#     return sample


# def test_logit_convergence(hld_sample, caplog):
#     """Test the GLM fit inside a function."""
#     # Enable logging capture to see the engine's internal logs
#     caplog.set_level(logging.DEBUG)

#     logger.info("Starting GLM Fit...")

#     try:
#         logit_model = hld_sample.glm.fit(
#             y="pov_status",
#             x=[
#                 "hhsize",
#                 "rooms",
#                 Cat("urbrur", ref="Rural")
#             ],
#             family="binomial",
#             link="logit",
#             max_iter=500,
#         )
#         assert logit_model is not None

#         # Verify coefficients exist
#         assert len(logit_model.coefs) > 0

#         # Check that the categorical variable was expanded correctly
#         # "urbrur" likely has levels like "Urban", "Rural".
#         # With ref="Rural", we expect "urbrur_Urban" (or similar) in the terms.
#         terms = [c.term for c in logit_model.coefs]
#         assert any("urbrur" in t for t in terms)

#     except Exception as e:
#         pytest.fail(f"Model failed to converge: {e}")
