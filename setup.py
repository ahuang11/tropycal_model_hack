from setuptools import setup

setup(name='tropycal_model_hack',
      version='0.0.1',
      description='Inserting model data into tropycal storm dict',
      url='https://github.com/ahuang11/tropycal_model_hack',
      packages=['tropycal_model_hack'],
      include_package_data=True,
      install_requires=[
                        'pandas',
                        'matplotlib',
                        'tropycal',
                        ],
      keywords=['tropycal', 'model', 'hack'],
      entry_points={
          'console_scripts': [
              'plot_cone = tropycal_model_hack.main:plot_forecast_cone_cli',
              'plot_global = tropycal_model_hack.main:plot_global_tracks_cli'
          ]
      },
      zip_safe=True)
