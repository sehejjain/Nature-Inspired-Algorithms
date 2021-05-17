from algorithms.algorithm import Algorithm
from algorithms.cuckoo_search import CuckooSearch
from algorithms.differential_evolution import DifferentialEvolution
from algorithms.fruitfly_optimization_algorithm import FruitFly
from algorithms.grey_wolf_optimizer import GreyWolfOptimizer
from algorithms.krill_herd import KrillHerdBase, KrillHerd
from algorithms.particle_swarm_optimization import ParticleSwarmOptimization
from algorithms.squirrel_search_algorithm import SquirrelSearchAlgorithm
from algorithms.water_wave_optimization import WaterWaveOptimization
from algorithms.whale_optimization_algorithm import WhaleOptimizationAlgorithm

__all__ = [
    'Abbreviation',
    'Algorithm',
    'CuckooSearch',
    'DifferentialEvolution',
    'FruitFly',
    'GreyWolfOptimizer',
    'KrillHerd',
    'ParticleSwarmOptimization',
    'SquirrelSearchAlgorithm',
    'WaterWaveOptimization',
    'WhaleOptimizationAlgorithm'
]

Abbreviation = {
    'CuckooSearch': 'CS',
    'DifferentialEvolution': 'DE',
    'FruitFly': 'FOA',
    'GreyWolfOptimizer': 'GWO',
    'KrillHerd': 'KH',
    'ParticleSwarmOptimization': 'PSO',
    'SquirrelSearchAlgorithm': 'SSA',
    'WaterWaveOptimization': 'WWO',
    'WhaleOptimizationAlgorithm': 'WOA',
}
