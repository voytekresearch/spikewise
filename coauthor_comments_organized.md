# Coauthor comments: extracted and organized

Source: `AP_waveform_empirical_paper_DRAFT (3).docx`

## Overview

- 69 coauthor comment threads/items: 67 from Ashley Juavinett and 2 from Ryan Hammonds.
- 2 replies from Blanca Martin are present in the file and summarized under the relevant Ryan items.
- The list below consolidates repeated comments into actionable tasks. Word comment IDs are retained in brackets for traceability.
- Comments that only express approval or document navigation are recorded separately and do not require action.

## Priority 1: analysis, interpretation, and claims

### Stimulation classifier and Figure 3

- Replace the single train/test-split confusion matrix with out-of-fold predictions so that all spikes are represented. Explain why the current matrix shows about 150 spikes while Supplementary Figure 4 suggests about 1,000. [Ryan 63]
- Explain the substantial constant-to-pink misclassification. Test/report whether low-amplitude constant-current sweeps, which are closer to the average pink-noise amplitude, are preferentially classified as pink. Add the supporting analysis and statistics to the supplement. [Ryan 63]
- Report exact p-values for the pairwise correlations in Figure 3E and clarify/apply correction for multiple comparisons. [Ashley 65]
- Clarify whether “spike waveform features” in the ridge model means all eight features. [Ashley 69]
- Consider whether multimodality should be quantified in the full feature space (Ashley suggests Mahalanobis distance) rather than declaring a cell multimodal when at least one feature clusters. This is a substantive suggestion, but may be optional if the current operational definition is well justified. [Ashley 38]

### LFP target variance and model validity

- Demonstrate that LFP targets have enough raw variation that regression performance cannot be obtained by predicting a near-constant value. Ryan suggests reporting target mean/variance and comparing the offset with coefficients. [Ryan 66]
- Existing response/analysis: because targets are z-scored, the fitted offset is not informative; raw targets are not nearly flat, and target variance is approximately uncorrelated or only weakly correlated with R² across pre/post windows and all four targets. Add this evidence to the main text or supplement, with numerical results rather than only links to plots. [Blanca 67]

### Claims that should be narrowed or better supported

- Remove or substantially soften the claim that LFP-derived predictors could reveal spike-level waveform dynamics where intracellular access is unavailable: the paper does not directly show that LFP predicts waveform shape. [Ashley 50]
- Reconsider the biological-computing/energy framing. The present results do not point to a solution for the energy problem; retain only a carefully bounded biological-realism implication if desired. [Ashley 56]
- Qualify the claim that binary outputs constrain “our” theoretical models (for example, “many” or “most” models), since not all computational neuroscience assumes binary outputs. [Ashley 5]
- Rework the claim that modern ANN architectures moved away from binary units because of superior performance. Clarify that they use continuous activation functions and replace LeCun (2015) with a citation that directly supports the binary-versus-continuous comparison, or remove the historical/performance claim. [Ashley 6–7]
- If stating that spikes are affected/determined by past and future spikes, choose the causal strength carefully and make clear which findings are from prior work versus the present study. [Ashley 16–18]
- Clarify the possible stimulation confound in spe-1 recordings. Ashley notes that imperfect bridge balance could produce a membrane-potential shift or small capacitance transients. Explain whether these signatures were checked or list this more explicitly as a limitation. [Ashley 52]

## Priority 2: manuscript structure and scientific clarity

### Abstract and terminology

- Replace vague “variance” with the measured quantity: field-potential/LFP amplitude variance or standard deviation. Decide whether the intended concept is statistical variance, standard deviation, or broader variability, and use that terminology consistently throughout the abstract, main text, methods, and captions. [Ashley 0–1, 34, 51, 68, 70]
- Consider revising “Instead” to: “Rather than signaling by changing amplitude, neurons increase the number of APs per unit time as a function of input intensity.” [Ashley 3]
- Consider replacing the broad phrase “changed neuroscience” with a more precise statement such as “changed how we conceptualize computation in the brain.” [Ashley 4]

### Introduction and motivation

- Strengthen the transition to Hodgkin–Huxley, possibly noting that the field has long known APs arise from variable, noisy, time-varying conductances. [Ashley 8–9]
- Remove the patch-clamp history sentence unless its direct benefit to this argument is made explicit. [Ashley 10]
- Add examples/citations showing that improved recordings revealed how ion-channel dynamics shape waveforms. Ashley suggested DOI `10.1093/cercor/bhh092` (mentioned twice). [Ashley 11–12]
- Cite Kilosort, preferably the 2024 Nature Methods paper, when discussing modern spike sorting. [Ashley 13–14]
- Replace “conventional” spike sorting with a precise term such as “clustering-based,” because template-based approaches are now conventional. [Ashley 15]
- Explicitly introduce prior work that predominantly uses AP width and amplitude; Ashley suggests a JNeurosci paper (`17/10/3425`). [Ashley 19]
- Replace the narrow “stimulation input” framing with a broader motivation involving ongoing network dynamics and brain states. [Ashley 20]
- Improve sentence order (“Using…, we first show…”). [Ashley 21]
- Do not redefine LFP after it has already been defined. [Ashley 22]
- Add motivation for testing whether peri-somatic LFP, described as a proxy for local ionic milieu, covaries with waveform shape. [Ashley 23]
- Introduce both datasets briefly before they are first used, including their names, recording types, and roles in the study. Move the current detailed Allen dataset description earlier and note that 200 kHz intracellular data are unusually rare, which helps explain why pvc-6 contains only two cells. [Ashley 26, 28, 30–31, 47]

### Results presentation

- Clarify whether “pvc-6” refers to the dataset, data, or cells. [Ashley 27]
- Prefer “how a single neuron’s AP waveform varied” over “within-neuron AP waveform…” where it improves readability. [Ashley 32]
- Move the Figure 3A citation to the sentence describing what the schematic actually depicts; the schematic itself does not demonstrate an effect. [Ashley 33]
- Revise “Spike waveform features were significantly correlated with all three stimulus properties” to avoid implying every feature correlates with every target; suggested wording: “Different spike waveform features were significantly correlated with the three stimulus properties.” [Ashley 35]
- Use consistent recording terminology (“intracellular” versus “patch”). [Ashley 37]
- Replace “ISI cluster” with an interpretable description such as “whether a cell had a short or long ISI”; consider “short/long” rather than “low/high.” [Ashley 39]
- Spell out normalized root mean squared error (nRMSE) at first use in the main text. [Ashley 40]
- Explain “temporal drift” explicitly (for example, waveform changes were not simply systematic changes over time) and add a concluding sentence. [Ashley 41]
- Explain earlier that the relevant metric captures waveform-shape differences independently of amplitude. [Ashley 42]
- Add a high-level conclusion to the within-versus-between-cell comparison: waveforms within one cell can be as different as, or more different than, waveforms from different cells. [Ashley 43]
- Move or remove the Neuropixels detail depending on where the dataset is introduced. [Ashley 44]
- Add a concluding sentence that waveform features reflect population activity beyond what is captured by raw spike rate. [Ashley 45]

### Discussion

- Consider “into how neural activity shapes behavior, and vice versa” in place of “into the mechanisms underlying behavior.” [Ashley 48]
- Consider adding the relevant Sabatini paper at the marked citation location. [Ashley 49]
- State what neuron types are represented in pvc-6 when discussing generalization across neuronal subtypes. [Ashley 53]
- Move, specify, or remove the vague sentence about future work on waveform variability, network dynamics, brain states, cognition, and behavior; it currently follows the cell-type discussion awkwardly. [Ashley 54]
- Remove the detailed sentence about how spiking neural networks are trained unless it is needed for the argument. [Ashley 55]

## Priority 3: figures and captions

### Figure 1

- Expand the caption so a reader can understand each schematic, especially the population-code panel. [Ashley 58]
- Make inhibitory versus excitatory cell types more explicit in the spike-sorting schematic. [Ashley 59]
- Replace “integrate with systems and cognitive neuroscience” with a concrete description involving other evidence streams, such as LFPs and inputs. [Ashley 60]

### Figure 2

- Choose a colorblind-safe palette that also remains interpretable in black and white, especially for panel A. [Ashley 61]
- Enlarge panel C and the time labels in panel F. [Ashley 62]

### Figure 3

- Redesign panel A so the arrow/question mark and black waveforms clearly communicate the research question rather than looking like a classification task. [Ryan 63]
- Update panel B to use out-of-fold predictions and discuss constant-versus-pink errors. [Ryan 63]
- Update panel E’s statistical reporting with exact p-values and multiple-comparison handling. [Ashley 65]

## Minor edits and wording suggestions

- Accept/review the small tracked-word edits around “by” and abbreviation usage. [Ashley 2, 25]
- Change “to infer” to “use these dynamics to infer” if it improves the sentence. [Ashley 29]
- Several comments request short closing or summary sentences; address these while restructuring rather than as isolated additions. [Ashley 41, 43, 45]

## Comments requiring no action

- Ashley marked the stimulus-results summary as strong. [Ashley 36]
- Ashley approved the explanation that the distance metric is amplitude-independent. [Ashley 42]
- Ashley liked the systems-level framing. [Ashley 46]
- The “MAIN FIGURES” header was added only for Google Docs navigation. [Ashley 57]
- Ashley’s parenthetical comment about jumping in the outline is navigational. [Ashley 24]

## Suggested order of work

1. Resolve terminology: variance, standard deviation, and variability.
2. Re-run/update Figure 3 analyses: out-of-fold confusion matrix, error analysis, exact p-values/multiple-comparison correction, and LFP-target variance checks.
3. Narrow unsupported LFP-prediction, ANN-performance, and energy-efficiency claims.
4. Reorganize the Introduction so datasets and motivation appear before first use.
5. Revise Results explanations and add the requested high-level conclusions.
6. Revise figures/captions for clarity, accessibility, and legibility.
7. Complete citations and minor wording edits.
