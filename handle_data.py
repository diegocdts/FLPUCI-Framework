from app_validation.script_for_infos import StrategyInfos
from instances.arguments import arguments
from utils.new_dataset import ExportTrace

args = arguments()

#info = StrategyInfos(args.dataset, args.approach, args.strategy.type, args.choice)
#info.get_community_id_maps()

et = ExportTrace(args.dataset)
et.write_trace()