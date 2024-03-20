from data_analysis.sample_distribution import SampleAnalysis, heatmap_matrix
from instances.data import sfc, rt, ngsim

analysis = SampleAnalysis(ngsim, interval=0)
pearson_matrix = analysis.pearson_correlation_at_interval()
kendal_matrix = analysis.kendaltau_correlation_at_interval()
heatmap_matrix(pearson_matrix, kendal_matrix)
