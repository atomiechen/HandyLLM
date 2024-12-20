import pytest


# def pytest_collection_modifyitems(config, items):
#     # enumerate all test items
#     for item in items:
#         # temporary workaround for respx and httpx compatibility issue
#         if "asyncio" in item.keywords:
#             # if the test item is asynchronous (marked with @pytest.mark.async),
#             # skip these tests
#             item.add_marker(
#                 pytest.mark.skip(
#                     reason="Skipping due to respx and httpx compatibility issue"
#                 )
#             )
