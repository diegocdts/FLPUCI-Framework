from app_validation.epidemic_analysis import EpidemicAnalysis
from instances.arguments import arguments

args = arguments()

epidemic_analysis = EpidemicAnalysis(args.dataset, args.approach, args.strategy.type, args.choice)
epidemic_analysis.analysis('/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/time_infection.txt')
