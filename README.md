# home-energy-management-system

This is the implementation for our paper: ["Multi-objective framework for a home energy management system with the integration of solar energy and an electric vehicle using an augmented ε-constraint method and lexicographic optimization"](https://doi.org/10.1016/j.scs.2022.104289), published at Sustainable Cities and Society.


## Setup 

```bash
conda env create -n hems --file env.yml
conda activate hems
```


## Structure

```bash
.
└── src/
    ├── components/
    │   ├── controllable_load.py
    │   ├── electric_vehicle.py
    │   ├── electric_water_heating.py
    │   ├── energy_storage.py
    │   ├── hvac_system.py
    │   ├── non_controllable_load.py
    │   ├── renewables.py
    │   └── utility_grid.py
    ├── utils/
    │   └── decision_making.py
    ├── config.py
    ├── hems.py
    └── main.py
```


## How to run

### Single-objective optimization

```
python3 main.py --mode single --obj energy_cost
```

### Multi-objective optimization

```
python3 main.py --mode multi --num_grid_points 6
```


## Citation
If you find the codes useful in your research, please consider citing:
```
@article{Huy2023,
   author = {Truong Hoang Bao Huy and Huy Truong Dinh and Daehee Kim},
   doi = {10.1016/j.scs.2022.104289},
   issn = {22106707},
   journal = {Sustainable Cities and Society},
   month = {1},
   pages = {104289},
   title = {Multi-objective framework for a home energy management system with the integration of solar energy and an electric vehicle using an augmented ε-constraint method and lexicographic optimization},
   volume = {88},
   year = {2023},
}
```

## License
[MIT LICENSE](LICENSE)