> As machine learning continues to permeate sensitive industries — from finance and healthcare to education and law — the demand to understand and *trust* the decisions made by complex models has become urgent. This is where *explainability* steps in. In this blog post, we’ll explore what explainability means in the context of machine learning, and how the [Alibi](https://github.com/SeldonIO/alibi) open-source library provides a rich suite of tools to help us peek under the hood of our predictive models.

**Docs:** [Alibi Explain — Alibi 0.9.7.dev0 documentation](https://docs.seldon.io/projects/alibi/en/latest/index.html)

## What Is Explainability?

In simple terms, explainability refers to the ability to describe and justify the outputs of machine learning models. As models become more complex and opaque — think deep neural networks — understanding why a certain prediction was made becomes critical.

Explainability lets us answer key questions like:

- How do predictions change with respect to input features?
- Which features were most important for a given prediction?
- What minimal change to inputs would flip a prediction?
- How much did each feature contribute to the prediction?

This knowledge not only builds **trust** and **transparency**, but it also opens doors to debugging, auditing, and improving model performance.

## Meet Alibi: Explainability in Practice

[Alibi](https://github.com/SeldonIO/alibi) is an open-source Python library built specifically to tackle explainability. It provides a wide range of *explainers* — algorithms that offer insight into your model’s behavior. Whether you’re working with tabular data, images, or text, and whether your model is a neural network or a random forest, Alibi has a tool that fits.

Each explainer in Alibi is tailored based on:

- **Data type** (e.g., tabular, image, text)
- **Model type** (e.g., black-box, white-box)
- **Task type** (e.g., classification, regression)

## Why Explainability Matters

1. **Trust & Ethics**
   For high-stakes applications like medical diagnosis or loan approval, we need to know *why* a model makes a decision. Explainability allows regulators and users to validate that the model is fair and justifiable.
2. **Debugging & Testing**
   When models fail or underperform, explainers help trace the issue back to data or architecture flaws. For example, we can verify if a model is relying on spurious correlations (like background color in images).
3. **Functionality Enhancements**
   Want to know how to change your model’s output? Some Alibi explainers help users *interact* with the model by showing how input tweaks can lead to desired outcomes.
4. **Research & Insight**
   Understanding the decision boundary of complex models helps researchers push the boundaries of what’s possible in machine learning — and do it responsibly.

## Black-box vs White-box Explainability

Alibi supports both **black-box** and **white-box** explainability:

- **Black-box explainers** don’t need access to the model’s internal logic. They only require input-output access, making them broadly applicable.
- **White-box explainers** can access gradients or internal parameters, and thus offer faster or more detailed insights — but only for compatible models like neural networks or decision trees.

> *📝 Note: In Alibi, “black-box” doesn’t mean “incomprehensible.” It just means the explainer works without peeking inside the model.*

## Local vs Global Insights

- **Local insights** focus on individual predictions. E.g., what specific features caused this loan application to be denied?
- **Global insights** focus on model behavior across the entire dataset. E.g., how does income affect predicted default rates overall?

Understanding both perspectives is crucial for comprehensive model evaluation.

## Types of Explainability Methods in Alibi

Here’s a categorized breakdown of some of Alibi’s most powerful tools:

Press enter or click to view image in full size

![img](https://miro.medium.com/v2/resize:fit:1102/1*DlD_TbrFLiI9FSo9-vtbAg.png)

## Caution: Interpretability ≠ Objectivity

While explainers enhance understanding, they are not free from bias:

- **Data Bias**: If your data contains biases (e.g., racial or gender skew), the model and its explanations will reflect them.
- **Human Bias**: Interpretations can be swayed by our own expectations. If a model’s reasoning doesn’t align with what we expect, we might distrust it even when it’s correct.
- **Complexity ≠ Incorrectness**: Some explanations (e.g., Anchors near decision boundaries) may be complex. Don’t mistake complexity for error.

> *🧠 Example: Risk analysts using SHAP values to validate a model may confirm that it “makes sense” — but only within the confines of their own (potentially biased) expectations.*

## Getting Started with Alibi

Alibi integrates easily into Python-based ML workflows. Here’s a minimal setup for using an explainer:

```
from alibi.explainers import KernelShap
import numpy as np

explainer = KernelShap(predict_fn=model.predict_proba, task='classification')
explainer.fit(X_train)
explanation = explainer.explain(X_instance)
print(explanation.shap_values)
```

Alibi supports other explainers like `AnchorTabular`, `IntegratedGradients`, and `Counterfactuals`, each tailored for specific use cases and model types.

## More CODE Examples

### 1. 📈 Local Feature Attribution: Integrated Gradients

```
from alibi.explainers import IntegratedGradients
import numpy as np

# Scaled input sample
x_input = scaler.transform(x)
# Initialize IG explainer
ig = IntegratedGradients(model,
                         layer=None,
                         method="gausslegendre",
                         n_steps=50,
                         internal_batch_size=100)
# Explain instance
result_ig = ig.explain(x_input, target=0)
# Visualize
plot_importance(result_ig.data['attributions'][0], features, '"good"'
```

### 2. 🤝 Local Attribution: Kernel SHAP (Neural Net)

```
from alibi.explainers import KernelShap

predict_fn = lambda x: model(scaler.transform(x))
explainer_shap = KernelShap(predict_fn, task='classification')
explainer_shap.fit(X_train[:100])
result_shap = explainer_shap.explain(x)
plot_importance(result_shap.shap_values[0], features, 0)
```

### 3. 🌳 Local Attribution: Kernel SHAP (Random Forest)

```
predict_fn_tree = lambda x: rfc.predict_proba(scaler.transform(x))
explainer_tree = KernelShap(predict_fn_tree, task='classification')

explainer_tree.fit(X_train[:100])
result_tree = explainer_tree.explain(x)
plot_importance(result_tree.shap_values[1], features, '"Good"')
```

### 4. ⚡ Fast Local Attribution: Interventional Tree SHAP

```
from alibi.explainers import TreeShap
expl_interv = TreeShap(rfc, model_output='raw', task='classification')
expl_interv.fit(scaler.transform(X_train[:100]))
result_interv = expl_interv.explain(scaler.transform(x), check_additivity=False)
plot_importance(result_interv.shap_values[1], features, '"Good"')
```

### 5. 🧭 Path Dependent Tree SHAP

```
path_expl = TreeShap(rfc, model_output='raw', task='classification')
path_expl.fit()
result_path = path_expl.explain(scaler.transform(x))
plot_importance(result_path.shap_values[1], features, '"Good"')
```

### 6. 🧱 Local Necessary Features: Anchors

```
from alibi.explainers import AnchorTabular

predict_fn = lambda x: model.predict(scaler.transform(x))
expl_anchor = AnchorTabular(predict_fn, features)
expl_anchor.fit(X_train, disc_perc=(25, 50, 75))
result_anchor = expl_anchor.explain(x, threshold=0.95)
print('Anchor =', result_anchor.data['anchor'])
print('Precision = ', result_anchor.data['precision'])
print('Coverage = ', result_anchor.data['coverage'])
```

### 7. 🌍 Global Attribution: ALE (Accumulated Local Effects)

```
from alibi.explainers import ALE
from alibi.explainers.ale import plot_ale

predict_fn = lambda x: model(scaler.transform(x)).numpy()[:, 0]
ale = ALE(predict_fn, feature_names=features)
exp_ale = ale.explain(X_train)
plot_ale(exp_ale, features=['alcohol'], line_kw={'label': 'Probability of "good" class'})
```

### 8. 🔁 Counterfactuals via Reinforcement Learning

```
from alibi.explainers import CounterfactualRL

predict_fn = lambda x: model(x)
cfrl_explainer = CounterfactualRL(
    predictor=predict_fn,
    encoder=ae.encoder,
    decoder=ae.decoder,
    latent_dim=7,
    coeff_sparsity=0.5,
    coeff_consistency=0.5,
    train_steps=10000,
    batch_size=100,
)
cfrl_explainer.fit(X=scaler.transform(X_train))
result_cfrl = cfrl_explainer.explain(X=scaler.transform(x), Y_t=np.array([1]))
print("Original prediction:", model.predict(scaler.transform(x))[0].argmax())
print("Counterfactual prediction:", model.predict(result_cfrl.data['cf']['X'])[0].argmax())
# Compare input and counterfactual
cf_instance = scaler.inverse_transform(result_cfrl.data['cf']['X'])
compare_instances(x, cf_instance)
```

- *Parts of this article were written using Generative AI*
- Subscribe/leave a comment if you want to stay up-to-date with the latest AI trends.