from app_validation.epidemic_analysis import EpidemicAnalysis
from app_validation.script_for_infos import StrategyInfos
from instances.arguments import arguments

args = arguments()

epidemic_analysis = EpidemicAnalysis(args.dataset, args.approach, args.strategy.type, args.choice)
epidemic_analysis.analysis('/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/time_infection.txt')

info = StrategyInfos(args.dataset, args.approach, args.strategy.type, args.choice)
info.get_community_id_maps()
info.get_previous_community_count()
