from app_validation.script_for_infos import StrategyInfos
from components.community_identification import CommunityIdentification
from instances.arguments import arguments
from utils.losses import replot_losses

args = arguments()

replot_losses(args.dataset.name, args.approach, args.dataset.proximal_term)

framework = CommunityIdentification(args.dataset,
                                    args.approach,
                                    args.properties,
                                    args.parameters,
                                    args.strategy,
                                    args.best_metric)
framework.model_training(args.first_interval, args.last_interval)
framework.compare_window_strategies()
framework.time_evolution(args.choice)

info = StrategyInfos(args.dataset, args.approach, args.strategy.type, args.choice)
info.get_community_id_maps()
info.get_previous_community_count()
