# 2023-SCS-HEMS

This is the implementation for our paper: ["Multi-objective framework for a home energy management system with the integration of solar energy and an electric vehicle using an augmented ε-constraint method and lexicographic optimization"](https://doi.org/10.1016/j.scs.2022.104289), published at Sustainable Cities and Society.

<!-- ## Environment 

- tensorflow: 2.0
- torch: 1.9 -->

<!-- ## Dataset
We opensource in this repository the model used for the ISO-NE test case. Code for ResNetPlus model can be found in /ISO-NE/ResNetPlus_ISONE.py

The dataset contains load and temperature data from 2003 to 2014. -->

## Structure

```bash
.
└── src/
    └── config.py
    └── decision_making.py
    └── hems.py
    └── main.py
    └── util.py
```

## How to run

### Single-objective optimization

```
python3 main.py --mode multi --num_grid_points 7
```

### Multi-objective optimization

```
python3 main.py --mode single --objective energy_cost --num_grid_points 7
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

<!-- ## License
[MIT LICENSE](LICENSE) -->