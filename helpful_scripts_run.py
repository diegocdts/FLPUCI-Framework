from app_validation.opportunistic_routing import RoutingMetricAnalysis
from helpful_scripts.new_dataset import NewDataset

"""
nd = NewDataset('helsinki', '/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/movement/mobility_generator_helsinki_MovementNs2Report.txt')
nd.export_raw_data()

nd = NewDataset('manhattan', '/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/movement/mobility_generator_manhattan_MovementNs2Report.txt')
nd.export_raw_data()
"""

rma = RoutingMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/manhattan')
rma.metrics()

rma = RoutingMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/helsinki')
rma.metrics()
