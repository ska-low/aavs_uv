from aa_uv.utils import get_resource_path, get_config_path

def test_resource_loading():
    print(get_resource_path('catch/me/if/you/can'))
    print(get_resource_path('config/aavs3/uv_config.yaml'))


def test_get_config():
    print(get_config_path('aavs2'))
    print(get_config_path('aavs3'))

if __name__ == "__main__":
    test_resource_loading()