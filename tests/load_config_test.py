from test_utils import TestUtils
test_utils_instance = TestUtils()



from config_loader import LoadConfig 

config = LoadConfig("./src/configs")

print(config)