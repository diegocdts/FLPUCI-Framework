from data_analysis.sample_distribution import SampleAnalysis, heatmap_matrix
from instances.data import sfc

analysis = SampleAnalysis(sfc, interval=0, threshold=0.7)
pearson_matrix = analysis.pearson_correlation_at_interval()
kendal_matrix = analysis.kendaltau_correlation_at_interval()
heatmap_matrix(pearson_matrix, kendal_matrix)
