from app_validation.opportunistic_routing import RoutingMetricAnalysis, IntraMetricAnalysis
from instances.arguments import arguments
from utils.new_dataset import ExportTrace

"""rm = RoutingMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace2/ONE/src/reports/manhattan')
rm.metrics()
rm.node_participation()"""

im = IntraMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace2/ONE/src/reports/intra/helsinki/',
                         [200,200,200], 2400)
im.probabilities()

im = IntraMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace2/ONE/src/reports/intra/manhattan/',
                         [400,400,400], 2400)
im.probabilities()

im = IntraMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace2/ONE/src/reports/intra/rt/',
                         [65,50,117,168,200,183], 14400)
im.probabilities()

im = IntraMetricAnalysis('/home/diegocdts/eclipse/eclipse-workspace2/ONE/src/reports/intra/sfc/',
                         [437,415,306,431,465,480], 14400)
im.probabilities()

"""args = arguments()
et = ExportTrace(args.dataset)
et.write_trace()"""