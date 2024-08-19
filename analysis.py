from app_validation.epidemic_analysis import EpidemicAnalysis
from app_validation.script_for_infos import StrategyInfos
from instances.arguments import arguments

args = arguments()

epidemic_analysis = EpidemicAnalysis(args.dataset, args.approach, args.strategy.type, args.choice)
epidemic_analysis.generate_time_infection('/home/diegocdts/eclipse/eclipse-workspace/One6/src/reports/epidemic modeling/',
                                          'epidemic_modeling_EventLogReport.txt')
epidemic_analysis.analysis()

info = StrategyInfos(args.dataset, args.approach, args.strategy.type, args.choice)
info.get_community_id_maps()
info.get_previous_community_count()
