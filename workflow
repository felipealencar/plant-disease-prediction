digraph {
	table_composition [label="Table composition"]
	dataset_analysis [label="Dataset analysis"]
	feature_engineering [label="Feature engineering"]
	regression_analysis [label="Regression analysis"]
	gan [label="Applying Generative Adversarial Networks (GAN)"]
	findings [label=Findings]
	table_composition -> dataset_analysis
	dataset_analysis -> feature_engineering
	feature_engineering -> regression_analysis
	regression_analysis -> gan
	gan -> findings
}
