# Beyond all-or-none: systematic variability in action potential waveforms

**Blanca Martin-Burgos³\*, Ashley Juavinett¹, Pamela D. Rivière¹, Ryan Hammonds², Bradley Voytek¹⁻⁴**

¹Department of Cognitive Science, ²Halıcıoğlu Data Science Institute, ³Neurosciences Graduate Program, ⁴Kavli Institute for Brain and Mind, University of California, San Diego, La Jolla, CA 92093

\*Correspondence: bcmartinb@gmail.com

---

**Data availability.** All code used for all analyses and plots are publicly available on GitHub at https://github.com/voytekresearch

**Acknowledgements.** Support: NIH National Institute of General Medical Sciences grant R01GM134363-01 (to B.V.). We thank Helpful Collaborator and Smart Labbies for their advice and feedback on the manuscript. Thanks to the data providers.

**Author contributions.** All authors conceived of the experiment(s) and developed the analyses, wrote analysis code, analyzed data, and wrote and edited the manuscript.

**Competing interests.** The authors declare no competing interests.

---

## Abstract

Action potentials are brief—approximately one millisecond—electrical impulses that form the mechanistic basis of how neurons communicate. While it is well known that the shape of action potentials can differ across neurons, the fundamental assumption is that the complex voltage waveforms of action potentials within a given neuron are reducible to binary spikes. This is codified as the All-or-none Law, introduced over a century ago. This assumption has constrained our conception of possible neural codes to those amenable to binary signaling, such as rate, temporal, and population codes. Here, we show that the All-or-none Law is incorrect. We first introduce an action potential parameterization method that we then applied to high temporal resolution (200 kHz) intracellular action potential recordings. We show that an action potential is not a digital '1', but is instead a rich signal whose fine-scale features influence the shape and timing of the next action potential and whose waveform is systematically biased by input drive. We then show that intracellular action potential waveforms cluster into distinct morphological groups within individual neurons recorded in intact circuits, and that these clusters are coupled to the concurrent extracellular local field potential—specifically to peri-spike LFP amplitude and aperiodic and gamma oscillatory dynamics. Pyramidal cells show significantly stronger LFP-spike waveform coupling than interneurons, and ISI-defined spike clusters predict post-spike aperiodic spectral dynamics in approximately 40% of cells. Our results have profound implications for systems and computational neuroscience and the development of biologically inspired artificial neural networks; dynamic action potential waveforms point to a broader landscape of possible neural codes, where neurons communicate with more than just binary spikes, but through their state-dependent waveform features.

---

## Introduction

Over 100 years ago, Lucas (1909) and Adrian (1914) performed seminal research demonstrating that the amplitude of a neuronal action potential (AP) is stable regardless of the intensity of the input stimulus. Instead, the number of APs that occur per unit time increases as a function of input intensity. These observations established the All-or-none law, which states that an AP either occurs or does not; it is a binary signal. This law fundamentally changed neuroscience. Under the assumption that this law is true, neural data analyses typically focus on identifying the rate at which APs occur or assessing the timing of these events in relation to other neural signals and behavioral states. This has constrained our theoretical models of neural coding to codes that are amenable to binary states such that the focus is on rate codes, temporal codes, phase codes, and population codes (Ferster & Spruston, 1995; Gerstner et al., 1997; Maass, 1998; Ahmed & Mehta, 2009), all of which disregard variation in spike waveform shape. Interestingly, while the original neurons that composed early artificial neural networks (ANNs) assumed binary firing states, modern ANN architectures have moved away from that assumption.

Although this reduction in complexity of AP waveforms to binary states is understandable from a historical perspective, where storing continuous electrophysiological time series sampled at 20 kHz or more was impossible, this signal reduction overlooks the fact that APs are fundamentally continuous, analog signals that are shaped by known biophysical mechanisms. For example, the development of patch-clamp electrophysiology enabled direct, high-resolution recordings of membrane voltage and ionic currents (Neher & Sakmann, 1976; reviewed in Verkhratsky & Parpura, 2014). As recording methodologies and resolution improved further, they provided insight into how ion channel dynamics shape spike waveform characteristics. These insights led to a growing recognition that distinct neuron types exhibit characteristic differences in AP shape.

One of the most widely replicated findings is the differentiation based on AP width between fast-spiking inhibitory interneurons, which exhibit narrower spikes, and excitatory pyramidal neurons, which exhibit broader ones (McCormick et al., 1985; Connors & Gutnick, 1990; Wilson et al., 1994; Brumberg et al., 1996; Rao et al., 1999; Frank et al., 2001). While AP width has been a reliable marker for distinguishing inhibitory and excitatory neurons, early research showed that other features also carry cell-type information, proposing a classification based on the product of rise time and half-height width (Kyriazi et al., 1996). More recent studies, using both intracellular and extracellular recordings, have expanded on this idea by combining multiple waveform features to classify a broader range of neuronal types and link them to functional properties (Bean, 2007; Trainito et al., 2019; Lee et al., 2023), gene expression profiles (Martini et al., 2023), physiological signals (Mosher et al., 2020), and neuronal morphology (Haynes et al., 2024; Moubarak et al., 2022).

The observation that AP waveforms differ systematically across neuron types laid the foundation for spike sorting methods, which use waveform shape to assign spikes recorded in extracellular recordings—a cornerstone of cognitive and systems neuroscience—to putative single units. These approaches rely on the assumption that each neuron generates a relatively consistent and distinguishable waveform such that clustering algorithms can separate overlapping activity from multiple nearby neurons and background noise (Lewicki, 1998). Building on this framework, a wide range of new algorithmic advances has improved accuracy, scalability, and compatibility with high-density and chronically implanted recording systems (Bestel et al., 2012; Su et al., 2013; Caro-Martín et al., 2018; Gibson et al., 2012; Buccino et al., 2022).

While AP waveform variability across neurons is well known, there is also substantial evidence from both intracellular and extracellular recordings showing that AP waveforms can vary systematically within a single neuron. The shape of an AP can shift dynamically with changing physiological conditions, dramatically impacting a neuron's effects on downstream targets and providing mechanisms of self-regulation to shape the neuron's responses to future inputs. A well-documented example is adaptation, in which action potentials broaden as a stimulus persists and firing rate decreases (Adrian & Zotterman, 1926; Aldrich et al., 1976; Jackson et al., 1991; Shao et al., 1999; Geiger & Jonas, 2000). This phenomenon is also observed in extracellular recordings, where the first spike in a burst is typically narrower and higher in amplitude than subsequent spikes (Kandel & Spencer, 1961).

When working to attribute APs from a bursty neuron, empiricists are generally well aware that APs corresponding to the first spike of each burst will tend to cluster together along waveform feature dimensions and will occasionally appear to cluster separately from the shorter and broader APs that correspond to subsequent spikes in the same burst. In this case, researchers typically attribute both clusters of APs to the same neuron rather than artificially "splitting" the neuron into two. However, it has recently been demonstrated that waveform-driven splitting can nonetheless occur in human single-unit recordings preceding and during ictal events (Merricks et al., 2021). In these recordings, AP waveforms changed so dramatically at seizure onset that conventional spike-sorting algorithms either assigned the altered waveforms to different neurons or discarded them entirely due to perceived instability. The interpretive consequences of artificial separations would suggest that one group of neurons is immediately suppressed as the seizure initiates while another ensemble of neurons quickly increases their firing as the event begins. Likewise, discarding spikes from neurons with unstable waveforms reduces statistical power and introduces bias, favoring neurons whose waveform properties remain artificially consistent across conditions.

While, as summarized above, there is ample evidence at the cellular and molecular levels that dynamic changes in ion channel function, synaptic input, and neuronal morphology are reflected in AP waveform shape, whether such variability extends to the systems level remains largely unexplored. Investigating the relationship between AP waveform variability and network dynamics or brain states could meaningfully extend conventional spike-based analyses and help link cellular mechanisms with systems-level processes, offering deeper insight into how neural activity supports cognition and behavior. This question can be approached by examining mesoscale signals such as the local field potential (LFP), which is widely used in systems and cognitive neuroscience to track brain and network states. The LFP reflects the aggregate of synaptic and intrinsic transmembrane currents within a local neuronal population (Buzsáki et al., 2012; Herreras, 2016), and it varies systematically with neuromodulatory tone, behavioral engagement, and sensory input (Donoghue et al., 1998; Liu & Newsome, 2006; Mazzoni et al., 2008; Lee & Dan, 2012; Prakash et al., 2022; Orellana et al., 2024). Prior work has shown that AP waveform shape can depend on membrane conductance in the milliseconds preceding spike initiation (de Polavieja et al., 2005), suggesting that features encoded in the LFP could be mechanistically linked to spike shape. If so, the LFP may reflect or co-vary with the physiological factors that shape spike waveforms, offering a powerful bridge between intracellular dynamics and the large-scale signals typically studied in systems and cognitive neuroscience.

Together, these findings help motivate a shift in how action potentials are conceptualized in systems and cognitive neuroscience (Figure 1). In these fields, action potentials have traditionally been conceptualized as binary events, supporting coding models based on spike timing and rate, while disregarding their waveform shape. However, advances in electrophysiological recordings and evidence have revealed that spike waveforms vary not only across neuron types but also within individual neurons over time, reflecting underlying physiological dynamics. While between-neuron differences in spike shape have long been leveraged for spike sorting and cell classification, within-neuron waveform variability remains underexplored at the systems level. Investigating how AP waveform shape relates to brain states, particularly through mesoscale signals such as the LFP, may provide new insight into the physiological basis of neural coding and cognition. Such an approach also carries important implications for spike sorting, which often assumes that neurons exhibit stable waveform signatures, potentially leading to misclassification when this assumption is violated.

To explore this possibility, we first devised a parameterization scheme to capture multiple features of AP waveform shapes. We then demonstrated that the properties of a time-varying current stimulation protocol causally influenced AP waveform shape parameters. We then extended this analysis to probe the relationship between intracellularly recorded AP waveforms and a neuron's ionic current milieu—as reflected in the peri-somatically recorded LFP—and AP waveform shape parameters. We show that within-neuron spike waveform variability is organized, LFP-coupled, and cell-type-dependent, with pyramidal cells exhibiting substantially stronger coupling between spike waveform clusters and concurrent network dynamics than interneurons.

---

## Approach

To investigate the functional significance of within-neuron AP waveform variability—and to bridge cellular-level mechanisms of spike generation with systems-level theories of neural coding—we developed a parameterization approach that quantifies fine-scale features of both extracellular and intracellular action potentials using an overcomplete set of waveform features. While most prior studies have focused exclusively on AP width and amplitude, our method is designed to capture the full complexity of spike shape, including ramp voltage, inflection amplitude and timing, peak width and sharpness, and repolarization decay dynamics. We are developing this method into an open-source software package (*spikeparam*; https://github.com/voytekresearch) that can be flexibly applied to both extracellular and intracellular recordings. Our goal is to promote consistency in AP feature extraction and enable reproducible, cross-study comparisons across a range of electrophysiological datasets.

To explore whether within-neuron AP waveform variability reflects differences in stimulation input or ongoing network dynamics and brain states, we leveraged two open-source datasets (Figure 2). We first demonstrated that AP features are systematically modulated by stimulation parameters using intracellular patch-clamp recordings with time-varying current injections. We then extended this analysis to a dataset of simultaneously recorded intracellular APs and local field potentials (LFPs), examining whether changes in the peri-somatic LFP—a proxy for local ionic milieu—co-vary with spike waveform shape.

---

## Results

### Single-unit waveform parameters are correlated

#### Parameterization captures intra-spike correlations

Using this rich parameterization method on intracellular patch-clamp recordings of APs (Figure 3A), we extracted a comprehensive set of waveform features, including ramp, inflection, and peak amplitudes; inflection time; peak width and sharpness; repolarization decay rate; and post-repolarization resting state amplitude. In addition, we computed the time to the next spike (inter-spike interval) and reported R² values for both the linear fit of the depolarization ramp and the exponential fit of the repolarization phase. From these extracted features, we observed within-neuron correlations (Figure 3B). For example, individual APs with sharper peaks exhibit faster repolarization, and APs that decay faster tend to be followed by shorter inter-spike intervals.

Given the strength of these intra-spike correlations, it may be possible to predict aspects of AP shape, such as decay dynamics, based on earlier waveform segments like the ramp or inflection point, and to infer the timing of the next spike. By changing specific waveform features, this method could be used in simulation to generate synthetic APs that reflect realistic physiological variability.

### Electrical stimulation causally influences spike waveform

#### Stimulation type shapes AP waveform morphology

To examine whether our parameterization methods captured AP waveform changes from inputs to the neurons, we used a dataset of intracellular recordings from mouse visual cortical neurons subjected to direct current stimulation (Berg, 2014). This dataset, recorded in vitro under whole-cell current clamp conditions and sampled at high resolution (200 kHz), allowed us to quantify fine-grained changes in spike shape across systematically varied input conditions.

We first examined how within-neuron AP waveform shape varied as a function of stimulation type (constant, ramp, or pink noise). We observed that the type of input significantly influenced the waveform structure (Figure 4A). Using a random forest classifier trained on our extracted AP features, we were able to predict stimulation type with high performance (Figure 4B: bootstrapped mean R²: 0.73; cross-validated mean accuracy: 74%). Among the most informative features for classification were peak amplitude and repolarization dynamics (Figure 4D).

#### Pink noise input statistics modulate AP features and predict firing timing

We next examined the effects of pink noise stimulation in particular, due to its dynamic fluctuation during recording and its spectral similarity to naturalistic 1/f neural activity, which may reflect fluctuations in brain state and underlying network dynamics (Gao et al., 2017; Medel et al., 2023). To characterize the stimulus, we computed the mean, variance, and spectral exponent of the injected current preceding each spike, and assessed how these properties related to AP waveform shape (Figure 5A). We observed that higher input amplitudes were associated with faster repolarization and shorter inter-spike intervals (Figure 5B). Using multivariate ridge regression, we found that AP waveform features—particularly peak sharpness and repolarization dynamics—explained 52% of the variance (bootstrapped mean R²: 0.52) in current amplitude (Figure 5C-D). In a complementary model, combining both AP and stimulus features, we could predict 22% of the variance (bootstrapped mean R²: 0.22) in time to the next spike (Figure 5E). These results suggest that stronger inputs not only increase firing rate (consistent with rate coding) but also dynamically modulate AP shape—specifically by accelerating repolarization, highlighting a possible role for input-driven K⁺ channel dynamics.

### Pre-spike local field potential state is coupled to spike waveform variability

To investigate whether AP waveform variability reflects broader network state dynamics, we analyzed the spe-1 dataset (Marques-Smith et al., 2018) of simultaneously recorded patch-clamp intracellular APs and extracellular LFPs recorded adjacent to the patched neuron in primary motor and somatosensory cortex in anesthetized rats (n = 43 cells; Figure 6A). This simultaneous recording design allowed us to ask, for the first time, whether spontaneous variation in AP waveform shape is organized by the concurrent local network state—and if so, which features of the LFP carry this information.

#### Spike waveform features cluster into distinct morphological groups within individual neurons

We first characterized the degree to which individual neurons' APs cluster into distinct waveform groups. Applying our parameterization to all 43 cells, we extracted spike features and asked, for each feature, whether the distribution was bimodal and whether the two resulting clusters differed meaningfully in waveform shape. We quantified cluster distinctiveness using normalized root mean square error (nRMSE) and cosine similarity between mean cluster waveforms—metrics that capture morphological rather than just statistical separation.

Across the population, log-transformed inter-spike interval (log ISI) was by far the most prevalent clustering feature, showing bimodal distributions in 33–38% of cells across metadata groups, and as high as 62.5% of dark neurons—roughly double or triple the prevalence of any other single spike feature. However, prevalence and waveform distinctiveness are not the same thing. Log ISI clusters occupy the low-nRMSE, high-cosine-similarity region of the waveform variance landscape: spikes split into ISI-history groups are not morphologically different from each other—the clustering is real but the waveform difference is minimal. The features with the largest waveform differences are peak width, peak sharpness, and inflection time, which show lower overall prevalence but sit at high nRMSE and low cosine similarity. When spikes cluster along these dimensions, the two groups are genuinely morphologically distinct in shape. This dissociation between prevalence and distinctiveness is an important one for interpreting what waveform variability means in different contexts.

A subset of seven priority cells—c3, c4, c21, c24, c26, c27, and c42—showed the largest mean waveform differences across features (mean nRMSE: 0.13–0.18), with the five highest-nRMSE cells all falling within this group. Top cell-feature pairs by combined nRMSE and cosine similarity difference included c24 × log ISI, c21 × peak width, c21 × peak amplitude, c27 × peak sharpness, and c42 × peak width. The remaining 36 cells showed near-zero waveform differences between clusters, consistent with statistical but not morphological within-neuron waveform variability across the majority of the population.

We also examined whether apparent clustering could be explained by temporal drift over the course of recordings—a potential confound in which gradual changes in patch quality produce time-ordered waveform changes that mimic biological clustering. We found that 87% of cell-feature pairs showed significant temporal drift in cluster membership (Spearman ρ ≠ 0, p < 0.05), with a striking feature-specific directionality: peak amplitude and sharpness consistently decreased over recording time (ρ ≈ −0.5), consistent with gradual seal degradation, while peak width, inflection time, and spike timing features increased monotonically (ρ ≈ +0.75). Critically, however, temporal drift magnitude was uncorrelated with waveform cluster distinctiveness—cells with significant drift showed statistically identical nRMSE and cosine similarity distributions to those without (Spearman ρ ≈ 0, ns; p_fdr > 0.7). The waveform cluster differences we observe are therefore not inflated by recording-time confounds, even though drift is pervasive. Only cortical depth weakly predicted drift magnitude (Spearman ρ = 0.17, p = 0.036); no other metadata variable predicted drift after FDR correction. The only metadata predictor of waveform cluster quality was EAP visibility: cells with a clearly visible extracellular AP on the Neuropixels channel showed lower cosine similarity between cluster waveforms (Spearman ρ = −0.381, p_fdr < 0.05), consistent with better signal isolation enabling detection of genuine morphological variability.

#### LFP amplitude and spectral state are broadly coupled to spike waveform clusters

To test whether spike waveform clusters reflect distinct network states, we used a time-resolved sliding-window analysis that extracted LFP features in windows around each spike and asked whether windows aligned with different waveform clusters showed significantly different LFP properties. We focused on five LFP features: LFP mean amplitude, aperiodic exponent, gamma-band area under the curve (AUC), theta-band AUC, and spectral goodness-of-fit (r²). Three features—aperiodic offset, LFP spectral exponent, and LFP standard deviation—showed 0% yield across all spike features and both cell subsets, confirming their redundancy; these were excluded from further analysis.

The results were striking in their consistency. LFP mean amplitude emerged as the dominant feature, showing near-universal coupling to spike waveform clusters: 100% of priority cells showed significant LFP amplitude differences between waveform cluster groups in at least one time window, as did 91% of all cells when clustering was done on log ISI (29 of 32 cells) and 100% of cells with spike-timing-based clusters. Effect sizes for LFP mean amplitude were also the largest of any LFP feature, with individual cells reaching Cohen's d up to 1.6 for log ISI clusters; most other features sat in the 0.1–0.4 range. Spectral features were also broadly coupled: aperiodic exponent showed significant modulation in 77–100% of cells depending on spike feature, gamma AUC in 70–100%, theta AUC in 60–92%, and spectral r² in 80–100%. Importantly, these high-yield findings were consistent across both the seven priority cells and the broader non-priority population, ruling out the possibility that results are driven solely by the subset of cells with the most extreme waveform variability.

Temporally, significant LFP-cluster windows were concentrated around the spike itself, with the highest density of significant windows at t ≈ 0 and tails extending approximately ±250 ms (Figure 6B). This timing pattern indicates that LFP-spike waveform coupling is primarily a peri-spike phenomenon rather than a reflection of sustained background state differences between cluster groups. Grand average LFP mean amplitude traces showed a consistent peri-spike dip for both cluster groups—differing in depth rather than shape—with higher-cluster-group spikes associated with deeper peri-spike LFP amplitude dips in cells where the effect was positive.

One important qualification is that while the yield of significant LFP-cluster coupling was near-universal, the direction of effects was heterogeneous across cells—in some neurons, higher-waveform-cluster spikes occurred during larger LFP amplitude modulations, while in others the relationship reversed. This directional heterogeneity was not explained by any measured metadata variable (all p_fdr > 0.1), suggesting it reflects cell-level biology or unmeasured factors such as recording geometry, projection target, or local circuit connectivity.

#### Pyramidal cells show stronger LFP-spike waveform coupling than interneurons

To determine whether LFP-spike coupling strength was modulated by cell type, we compared LFP effect sizes (|Cohen's d|) between putative pyramidal cells (PCs; n = 38) and interneurons (INs; n = 5). After Benjamini-Hochberg FDR correction, PCs showed significantly larger coupling than INs for four of five LFP features: aperiodic exponent (p_fdr = 0.003), spectral r² (p_fdr = 0.006), gamma AUC (p_fdr < 0.001), and LFP mean amplitude (p_fdr = 0.002). Theta AUC did not reach significance (p_fdr = 0.121–0.171). This cell-type difference replicated across both the priority cell subset and the full population. Critically, it was not explained by confounds in recording modality, clamp mode, dark neuron status, EAP visibility, or cortical depth—none of which predicted LFP coupling strength after FDR correction (all p_fdr > 0.20). The finding that PCs show consistently stronger broadband and gamma-band LFP-waveform coupling is consistent with their larger dendritic arbors and greater synaptic convergence, which may amplify the influence of network oscillatory state on the subthreshold membrane dynamics that shape AP morphology.

#### ISI cluster membership predicts post-spike aperiodic spectral dynamics

Finally, we examined whether spike waveform clusters were associated with systematic differences in LFP spectral state in fixed time windows before (−0.5 to −0.05 s) and after (+0.05 to +1.0 s) each spike. In contrast to the sliding-window analysis, the pre-spike window showed largely null results: no spike feature showed consistent pre-spike LFP state differences between clusters across the population (effects small, |g| < 0.2, centered at zero), suggesting that spike waveform cluster identity is not simply inherited from a sustained, pre-existing network state.

The post-spike window, however, revealed a specific and replicable coupling. Log ISI clusters predicted post-spike aperiodic exponent in 42% of all cells (15 of 36), with nearly identical yield across priority cells (43%; 3 of 7) and non-priority cells (41%; 12 of 29)—the consistency across independent subsets supporting the credibility of this finding. Log ISI also showed elevated yield for post-spike spectral r² (31% of all cells) and gamma AUC (22%). The general pattern within each cluster group was an increase in aperiodic exponent after spike events—a post-spike spectral steepening—alongside a decrease in gamma power, consistent with a known post-spike hyperpolarization or network state reset. The between-cluster difference in the magnitude of this post-spike exponent shift is what drives the 42% yield.

This post-spike ISI-cluster effect—absent from the pre-spike window—suggests that different ISI histories may exert a measurable influence on the subsequent evolution of local network spectral dynamics, rather than simply reflecting a pre-existing state. The effect is not universal: the approximately 60% of cells that do not show it are indistinguishable from those that do based on any available metadata variable (all p_fdr > 0.1), pointing to unmeasured cell-level factors—such as firing rate regime, projection target, or local circuit connectivity—as potential modulators.

---

## Discussion

Here we show that within-neuron AP waveform variability is systematic, organized, and coupled to the concurrent state of the local network. Using a novel parameterization approach applied to two open-source electrophysiological datasets, we demonstrate three complementary lines of evidence against the all-or-none conceptualization of action potentials. First, AP waveform features are not simply random or noisy: they are correlated with one another and with the timing of subsequent spikes, and they are systematically shaped by the character of the preceding input drive. Second, within individual neurons recorded in intact circuits, spike waveforms naturally segregate into distinct morphological clusters—particularly based on ISI history, peak width, and sharpness—and these clusters are broadly coupled to concurrent LFP amplitude and spectral state. Third, the strength of this LFP-waveform coupling is cell-type-dependent: pyramidal cells show significantly stronger coupling than interneurons across broadband amplitude and gamma-band features.

The finding that AP waveform features are modulated by stimulation parameters (pvc-6 dataset) provides a clear demonstration that input drive shapes spike morphology in real time. The random forest classifier's ability to decode stimulation type from AP features alone (bootstrapped R²: 0.73; cross-validated accuracy: 74%) shows that this encoding is not subtle—waveform shape carries substantial information about the input state of the neuron. The repolarization phase emerged as a dominant informative feature, consistent with the biophysical understanding that K⁺ channel dynamics are highly sensitive to recent voltage and conductance history (Geiger & Jonas, 2000; Shao et al., 1999). The correlation between pre-spike input amplitude and repolarization speed in the pink noise condition (R²: 0.52) further establishes a link between input statistics and the temporal structure of individual spikes, extending de Polavieja et al. (2005)'s demonstration that subthreshold conductance history shapes AP waveform into a quantitatively precise regime.

The spe-1 results reveal that this input-driven waveform variability extends to spontaneous, naturalistic firing in an intact circuit context. The near-universal coupling between peri-spike LFP amplitude and spike waveform clusters (100% yield in priority cells; 91% for log ISI across all cells) is striking, and stands in contrast to the pre-spike analysis, which found largely null effects. This pre/post dissociation indicates that the LFP-waveform relationship is event-locked rather than a reflection of a pre-existing sustained state, suggesting that what the LFP captures around the moment of a spike—the aggregate local synaptic drive and intrinsic currents—is meaningfully coupled to how that spike unfolds. This is consistent with theoretical and experimental evidence that membrane conductance in the milliseconds before spike initiation shapes waveform features (de Polavieja et al., 2005), and with the known sensitivity of repolarization kinetics to recent voltage history (Geiger & Jonas, 2000).

An important caveat is that the direction of LFP-waveform coupling is heterogeneous across cells. The near-universal yield of significant coupling is not accompanied by a consistent directional signature at the population level—in some neurons, higher-cluster-group spikes occur during larger peri-spike LFP amplitude modulations, while in others the relationship is reversed. This heterogeneity was not explained by any measured metadata variable, and likely reflects idiosyncratic differences in local circuit connectivity, recording geometry, or neuronal functional state. Future work with larger and more deeply characterized samples—including laminar position, projection targets, and behavioral state—will be needed to resolve the sources of this directional variance. Nonetheless, the near-universal yield of significant coupling, regardless of direction, firmly establishes that the relationship between peri-spike network state and spike morphology is a robust population-level phenomenon.

The cell-type analysis adds an important structural dimension to this picture. The finding that pyramidal cells show significantly stronger LFP-spike waveform coupling than interneurons—for aperiodic, gamma-band, and amplitude features—is consistent with the broader electrophysiology literature, in which LFP signals are thought to predominantly reflect synaptic currents impinging on pyramidal cell dendrites (Buzsáki et al., 2012). Pyramidal cells, with their extensive dendritic trees, integrate synaptic input across a large spatial volume, making their subthreshold membrane dynamics—and thus their spike morphology—more susceptible to fluctuations in the local field than the more compact interneurons. The specific absence of a cell-type effect for theta-band AUC may reflect the larger spatial scale over which theta oscillations operate relative to gamma or aperiodic fluctuations, but this warrants further investigation.

The post-spike analysis reveals that ISI history is a particularly interesting axis of waveform variability. Log ISI clusters predict post-spike aperiodic spectral steepening in approximately 40% of cells—a finding that is consistent across priority and non-priority subsets, but heterogeneous at the cell level. The post-spike—not pre-spike—nature of this effect is suggestive: rather than a pre-existing state driving both the ISI history and the subsequent waveform clustering, it may be that the firing pattern itself—mediated through K⁺ afterhyperpolarization, synaptic feedback, or other mechanisms—influences the subsequent evolution of local network spectral dynamics in an ISI-dependent fashion. This interpretation is necessarily preliminary given the observational nature of these data, but it opens a compelling direction for causal investigation using optogenetic or pharmacological manipulation of firing rate.

In addition to informing spike sorting protocols, capturing AP waveform variation and identifying its causal drivers may offer as-yet-untapped insights into the mechanisms underlying behavior. At the molecular level, modulation or deletion of specific potassium channels has been shown to alter spike shape and support activity-dependent plasticity (Hoppa et al., 2014; Richardson et al., 2022). These dynamic changes in ion channel function are influenced by neuromodulation (Qian & Saggau, 1999; Geiger & Jonas, 2000; Cantrell & Catterall, 2001; Bean, 2007), synaptic input and recent membrane voltage history (de Polavieja et al., 2005), and morphological features of axons and dendrites (López-Jury et al., 2018). Moreover, AP waveform shape—not just its rate or timing—can directly affect downstream neuronal activity: increasing the width or duration of an AP at presynaptic terminals enhances both the amplitude and duration of excitatory postsynaptic potentials in target neurons (Geiger & Jonas, 2000; Engel & Jonas, 2005; Shu et al., 2006; Roshchin et al., 2018). These findings suggest that dynamic variation in spike waveform may serve as a mechanism for fine-tuned regulation of synaptic efficacy and, ultimately, behavioral output.

Our findings also have direct implications for spike sorting. The assumption that a neuron emits a stable, consistent waveform—central to most existing spike-sorting algorithms—is violated in the vast majority of cells we examined: 87% of cell-feature pairs show significant temporal drift in waveform cluster membership. While we show that this drift does not inflate apparent cluster differences, it does mean that the waveform space occupied by a single neuron shifts systematically over the course of a recording. Algorithms that treat waveform variability as noise rather than signal may misclassify or discard spikes at the boundaries of this shift, introducing systematic bias in the recorded neural representation. Our parameterization approach offers a natural framework for characterizing and controlling for this variability in sorting pipelines, and the dissociation between temporal drift and waveform distinctiveness provides principled guidance for disentangling recording confounds from genuine biological variability.

Together, these results establish that AP waveforms are dynamic, state-dependent signals rather than stereotyped binary events. The all-or-none conceptualization of APs, while historically useful, has constrained our models of neural coding to a binary space that does not capture the full dimensionality of information available in spiking signals. Dynamic AP waveforms point to a broader landscape of possible neural codes—ones in which the content of a spike extends beyond its occurrence time to include the morphological features of the waveform itself, features that are systematically coupled to the network state in which the spike is embedded. Incorporating this dimension into systems and computational models of neural coding and cognition is a natural and necessary next step.

---

## Methods

### Open data sources

**Dataset 1: pvc-6 (controlled stimulation, mouse visual cortex).** In vitro whole-cell current clamp recordings were obtained from neurons in the adult mouse visual cortex. The dataset was downloaded from the Collaborative Research in Computational Neuroscience (CRCNS) database (Berg, 2014; http://dx.doi.org/10.6080/K0H12ZXD), which additionally describes the data acquisition protocol in detail. Recordings were sampled at 200 kHz and are available for two neurons, one of which was a molecularly confirmed somatostatin (SST)-positive visual cortical neuron. Each neuron received current injections in the form of short (3 ms) and long (500 ms) square pulses (constant current), linearly increasing current (ramp), and three variants of pink noise (1/f) current stimulation. For one neuron, pink noise current injections were scaled to 100%, 125%, and 150% relative to the intensity of the 500 ms square pulse. For the SST neuron, pink noise stimulation was scaled to 75%, 100%, and 125% of the intensity of the 500 ms square pulse.

**Dataset 2: spe-1 (simultaneous LFP and intracellular recording, rat cortex).** Complementary metal-oxide-semiconductor (CMOS) Neuropixels probes and patch-clamp recordings of cortical neurons were used to acquire simultaneous local field potential (LFP) and intracellular neuronal activity in anesthetized rats. This dataset was recorded and organized by Marques-Smith et al. (preprint) and is openly available on CRCNS (spe-1; http://dx.doi.org/10.6080/K0J67F4T) as well as a public Google Drive folder (http://bit.ly/paired_recs). Details regarding data access, loading, and processing are available at https://github.com/kampff-lab. Neuropixels signals were acquired at 30 kHz, with action potential and LFP bands digitized separately (AP band at 30 kHz, LFP band at 2.5 kHz). Patch-clamp signals were acquired at approximately 50.023 kHz (exact rates differed per cell, ranging from 50,023.87 to 50,023.92 Hz). The majority of recordings were performed in current-clamp, cell-attached (juxtacellular) configuration, though a subset used whole-cell current-clamp or voltage-clamp configuration.

**Cell selection (spe-1).** A total of 43 neurons were included in the dataset. Recording regions spanned primary motor cortex and primary somatosensory cortex (forelimb, hindlimb, and trunk representations). Cell types were classified as putative pyramidal cells (PC; n = 38) or interneurons (IN; n = 5: cells c7, c8, c12, c16, c18, and c22), based on extracellular AP waveform morphology (trough-to-peak and half-width duration). Cortical depth was estimated from the Neuropixels channel position and ranged from approximately 389 to 1,754 µm. For each cell, the Neuropixels channel closest to the recorded neuron was selected based on the channel that maximized peak-to-peak extracellular AP amplitude. Cells were selected for LFP analysis based on patch-seal quality and the availability of a Neuropixels electrode channel that captured the extracellular instantiation of the corresponding intracellularly recorded action potential. A subset of seven priority cells (c3, c4, c21, c24, c26, c27, c42) were identified post hoc as showing the strongest spike waveform cluster differences (highest mean nRMSE across features) and were used for detailed peri-spike LFP analyses.

### Signal preprocessing

**pvc-6.** Recordings were loaded directly from CRCNS in their provided format. No additional filtering was applied to the patch-clamp signal prior to spike detection, as recordings were already at 200 kHz with a high signal-to-noise ratio. Spike times and stimulus epoch labels (constant, ramp, pink noise) were extracted from the stimulus channel.

**spe-1.** Raw binary recordings were loaded using custom Python routines. Each signal type was bandpass filtered using a 4th-order Butterworth filter applied in zero-phase mode (forward and backward pass; scipy.signal.sosfiltfilt) with the following frequency bands: patch-clamp signal [10–25,000 Hz]; Neuropixels raw signal [300–14,000 Hz]; LFP signal [0.1–250 Hz]. Neuropixels data were stored as 16-bit integers and reshaped to (384 channels × samples) in Fortran order; the single channel corresponding to each recorded neuron was extracted using the per-cell channel mapping. Filtered signals were saved as NumPy (.npy) arrays for downstream analysis.

### Spike detection

Action potentials were detected from the patch-clamp signal using per-cell amplitude thresholds (pvc-6: derived from the stimulus protocol; spe-1: per-cell thresholds ranged from 0.81 to 137.2 mV). Spike times were defined as the index of threshold crossing, and each spike waveform was extracted as a fixed-length window centered around the waveform peak. Duplicate detections within a refractory period were removed. Spikes were aligned to their peak (maximum voltage) prior to parameterization.

### Action potential waveform parameterization

We developed a novel parameterization approach (implemented in the open-source *spikeparam* Python package; https://github.com/voytekresearch) that quantifies fine-scale features of action potential waveforms using an overcomplete set of intra-spike and inter-spike features. For each spike waveform, we first computed the first derivative and smoothed it using locally weighted scatterplot smoothing (LOWESS; smoothing fraction = 0.008; statsmodels.nonparametric.lowess). The peak index was defined as the sample with maximum voltage. The inflection point was detected by fitting two piecewise linear segments to the smoothed derivative in the pre-peak window, and finding the intersection of the two fitted lines. The ramp start was defined as 1 ms before the inflection point. Rise and decay half-amplitude midpoints were computed from the voltage trace as the samples where the voltage was halfway between the inflection amplitude and the peak amplitude, on the rising and falling flanks respectively. The exponential window started 2 ms after the peak and extended for 5 ms.

Intra-spike features extracted from each waveform included: ramp amplitude (slope of a linear fit to the 1 ms pre-inflection voltage window, mV/ms); inflection time (time of inflection point relative to peak, ms); inflection amplitude (membrane voltage at inflection point, mV); peak amplitude (maximum voltage, mV); peak width (time between rise and decay half-amplitude midpoints, ms); peak sharpness (mean voltage drop at ±0.1 ms around the peak, mV); exponential amplitude (A), exponential decay rate (exp_lambda, λ), and exponential constant (exp_const, C), from a three-parameter exponential decay model V(t) = A · exp(−λ · t) + C fit using scipy.optimize.curve_fit. Goodness of fit was assessed using R² for both the linear ramp fit and the exponential decay fit. Inter-spike features included the log-transformed inter-spike interval (log ISI): the log of the time elapsed from each spike to the subsequent spike (ms).

### Spike waveform clustering (spe-1)

For each cell and each spike feature, we assessed whether the within-neuron feature distribution was bimodal using a Gaussian mixture model with two components. Clustering quality was quantified using normalized root mean square error (nRMSE) and cosine similarity between the mean waveforms of the two cluster groups—metrics that capture morphological rather than statistical separation. Temporal drift was assessed using Spearman correlation between spike index (recording time order) and cluster label. Seven priority cells (c3, c4, c21, c24, c26, c27, c42) were identified as those with the highest mean nRMSE across spike features.

### LFP spectral analysis

Power spectral density (PSD) estimates for LFP windows were computed using the multitaper method (Babadi & Brown, 2014), which balances the bias–variance tradeoff for short-duration signals. We applied spectral parameterization (specparam/fooof; Donoghue et al., 2020) to extract periodic and aperiodic components from each PSD. Aperiodic features (offset, exponent) and periodic peak features (center frequency, power, bandwidth) were extracted per window. Gamma-band (30–80 Hz) and theta-band (4–8 Hz) power were quantified as area under the curve (AUC) within the respective frequency ranges.

### Sliding-window LFP-spike analysis

For each cell, LFP features were computed in sliding time windows around each spike event (window duration: 0.5 s, step: 10 ms). Each spike was aligned to its corresponding LFP window, and LFP feature distributions were compared between spike waveform cluster groups using Cohen's d as the effect size. Significant windows were identified using Welch's t-test or F-test (vectorized), with significance defined at p < 0.05. Population-level robustness was assessed using bootstrap confidence intervals (n = 1000 resamplings of cells). Within-cell specificity was confirmed using permutation tests (n = 500 label shuffles). Yield was defined as the percentage of cells showing at least one significant window for a given spike feature × LFP feature combination. Three LFP features (aperiodic offset, LFP spectral exponent, LFP standard deviation) showed 0% yield across all spike features and both cell subsets and were excluded from further analysis.

### Pre/post spike LFP analysis

For each spike, we extracted pre-spike (−0.5 to −0.05 s) and post-spike (+0.05 to +1.0 s) LFP windows and computed spectral features (aperiodic exponent, gamma AUC, theta AUC, r², LFP mean amplitude) for each window. We compared these features between spike cluster groups using Hedges' g as the effect size. A cell was counted as showing a significant pre- or post-spike effect if Hedges' g differed significantly from zero after permutation testing.

### Predictive models (pvc-6)

Stimulation type decoding was performed using three classifiers: logistic regression, support vector machine (SVM), and random forest, each trained and evaluated using stratified k-fold cross-validation. Classifier accuracy was bootstrapped (n = 1000 train/test splits) to generate distributions of performance. Feature importance was extracted from the trained random forest model. For the pink noise analyses, ridge regression (sklearn.linear_model.Ridge) was used to predict stimulation amplitude and inter-spike interval from AP feature vectors. Bootstrapped R² distributions were generated using n = 1000 resamplings of the training set.

### Statistical analyses

All data were analyzed in Python. Statistical analyses were performed using Scipy (Virtanen et al., 2020) and scikit-learn (Pedregosa et al., 2011), with electrophysiological data analyses performed using the neurodsp (Cole et al., 2019), fooof/specparam (Donoghue et al., 2020), and bycycle (Cole & Voytek, 2019) toolboxes. For group comparisons, we used Mann-Whitney U or Kruskal-Wallis tests (non-parametric). For correlations, we used Spearman rank correlation. Multiple comparisons were corrected using Benjamini-Hochberg false discovery rate (BH-FDR) correction throughout. Effect sizes are reported as Cohen's d (pooled standard deviation) for between-group comparisons and Spearman ρ for correlations. Additional analyses were performed using custom Python scripts, all of which are available on the laboratory GitHub account (https://github.com/voytekresearch).

---

## References

Adrian, E. D. (1914). The all-or-none principle in nerve. *The Journal of Physiology*, 47(6), 460–474.

Adrian, E. D., & Zotterman, Y. (1926). The impulses produced by sensory nerve endings. *The Journal of Physiology*, 61(4), 465–483.

Ahmed, O. J., & Mehta, M. R. (2009). The hippocampal rate code: anatomy, physiology and theory. *Trends in Neurosciences*, 32(6), 329–338.

Aldrich, R. W., Getting, P. A., & Thompson, S. H. (1979). Mechanism of frequency-dependent broadening of molluscan neurone soma spikes. *The Journal of Physiology*, 291(1), 531–544.

Babadi, B., & Brown, E. N. (2014). A review of multitaper spectral analysis. *IEEE Transactions on Biomedical Engineering*, 61(5), 1555–1564.

Bean, B. P. (2007). The action potential in mammalian central neurons. *Nature Reviews Neuroscience*, 8(6), 451–465.

Berg, J. (2014). In vitro whole-cell patch clamp recordings from visual cortex neurons in the adult mouse. CRCNS.org. http://dx.doi.org/10.6080/K0H12ZXD

Bestel, R., Daus, A. W., & Thielemann, C. (2012). A novel automated spike sorting algorithm with adaptable feature extraction. *Journal of Neuroscience Methods*, 211(1), 168–178.

Brumberg, J. C., Pinto, D., & Simons, D. J. (1996). Spatial gradients and inhibitory summation in the rat whisker barrel system. *Journal of Neurophysiology*, 76(1), 130–140.

Buccino, A. P., Garcia, S., & Yger, P. (2022). Spike sorting: new trends and challenges of the era of high-density probes. *Progress in Biomedical Engineering*, 4(2), 022005.

Buzsáki, G., Anastassiou, C. A., & Koch, C. (2012). The origin of extracellular fields and currents—EEG, ECoG, LFP and spikes. *Nature Reviews Neuroscience*, 13(6), 407–420.

Cantrell, A. R., & Catterall, W. A. (2001). Neuromodulation of Na+ channels: an unexpected form of cellular plasticity. *Nature Reviews Neuroscience*, 2(6), 397–407.

Caro-Martín, C. R., Delgado-García, J. M., Gruart, A., & Sánchez-Campusano, R. (2018). Spike sorting based on shape, phase, and distribution features, and K-TOPS clustering with validity and error indices. *Scientific Reports*, 8(1).

Cole, S. R., & Voytek, B. (2019). Cycle-by-cycle analysis of neural oscillations. *Journal of Neurophysiology*, 122(2), 849–861.

Cole, S., Donoghue, T., Gao, R., & Voytek, B. (2019). NeuroDSP: A package for neural digital signal processing. *Journal of Open Source Software*, 4(36), 1272.

Connors, B. W., & Gutnick, M. J. (1990). Intrinsic firing patterns of diverse neocortical neurons. *Trends in Neurosciences*, 13(3), 99–104.

de Polavieja, G. G. (2005). Stimulus history reliably shapes action potential waveforms of cortical neurons. *Journal of Neuroscience*, 25(23), 5657–5665.

Donoghue, J. P., Sanes, J. N., Hatsopoulos, N. G., & Gaál, G. (1998). Neural discharge and local field potential oscillations in primate motor cortex during voluntary movements. *Journal of Neurophysiology*, 79(1), 159–173.

Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., ... & Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655–1665.

Engel, D., & Jonas, P. (2005). Presynaptic action potential amplification by voltage-gated Na+ channels in hippocampal mossy fiber boutons. *Neuron*, 45(3), 405–417.

Ferster, D., & Spruston, N. (1995). Cracking the neuronal code. *Science*, 270(5237), 756–757.

Frank, L. M., Brown, E. N., & Wilson, M. A. (2001). A comparison of the firing properties of putative excitatory and inhibitory neurons from CA1 and the entorhinal cortex. *Journal of Neurophysiology*, 86(4), 2029–2040.

Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage*, 158, 70–78.

Geiger, J. R. P., & Jonas, P. (2000). Dynamic control of presynaptic Ca2+ inflow by fast-inactivating K+ channels in hippocampal mossy fiber boutons. *Neuron*, 28(3), 927–939.

Gerstner, W., Kreiter, A. K., Markram, H., & Herz, A. V. M. (1997). Neural codes: firing rates and beyond. *Proceedings of the National Academy of Sciences*, 94(24), 12740–12741.

Gibson, S., Judy, J. W., & Markovic, D. (2012). Spike sorting: the first step in decoding the brain. *IEEE Signal Processing Magazine*, 29(1), 124–143.

Haynes, V. R., Zhou, Y., & Crook, S. M. (2024). Discovering optimal features for neuron-type identification from extracellular recordings. *Frontiers in Neuroinformatics*, 18.

Herreras, O. (2016). Local field potentials: myths and misunderstandings. *Frontiers in Neural Circuits*, 10.

Hoppa, M. B., Gouzer, G., Armbruster, M., & Ryan, T. A. (2014). Control and plasticity of the presynaptic action potential waveform at small CNS nerve terminals. *Neuron*, 84(4), 778–789.

Jackson, M. B., Konnerth, A., & Augustine, G. J. (1991). Action potential broadening and frequency-dependent facilitation of calcium signals in pituitary nerve terminals. *PNAS*, 88(2), 380–384.

Kandel, E. R., & Spencer, W. A. (1961). Electrophysiology of hippocampal neurons. II. After-potentials and repetitive firing. *Journal of Neurophysiology*, 24(3), 243–259.

Kyriazi, H. T., Carvell, G. E., Brumberg, J. C., & Simons, D. J. (1996). Quantitative effects of GABA and bicuculline methiodide on receptive field properties of neurons in real and simulated whisker barrels. *Journal of Neurophysiology*, 75(2), 547–560.

Lee, K., Carr, N., Perliss, A., & Chandrasekaran, C. (2023). WaveMAP for identifying putative cell types from in vivo electrophysiology. *STAR Protocols*, 4(2), 102320.

Lee, S.-H., & Dan, Y. (2012). Neuromodulation of brain states. *Neuron*, 76(1), 209–222.

Lewicki, M. (1998). A review of methods for spike sorting: the detection and classification of neural action potentials. *Network: Computation in Neural Systems*, 9(4), R53–R78.

Liu, J., & Newsome, W. T. (2006). Local field potential in cortical area MT: stimulus tuning and behavioral correlations. *Journal of Neuroscience*, 26(30), 7779–7790.

López-Jury, L., Meza, R. C., Brown, M. T. C., Henny, P., & Canavier, C. C. (2018). Morphological and biophysical determinants of the intracellular and extracellular waveforms in nigral dopaminergic neurons. *The Journal of Neuroscience*, 38(38), 8295–8310.

Lucas, K. (1909). The "all or none" contraction of the amphibian skeletal muscle fibre. *The Journal of Physiology*, 38(2–3), 113–133.

Maass, W. (1998). A simple model for neural computation with firing rates and firing correlations. *Network: Computation in Neural Systems*, 9(3), 381–397.

Marques-Smith, A., Neto, J. P., Lopes, G., Nogueira, J., Calcaterra, L., Frazão, J., Kim, D., Phillips, M. G., Dimitriadis, G., & Kampff, A. R. (2018). Recording from the same neuron with high-density CMOS probes and patch-clamp: a ground-truth dataset and an experiment in collaboration. *bioRxiv*. https://doi.org/10.1101/370080

Martini, L., Amprimo, G., Di Carlo, S., Olmo, G., Ferraris, C., Savino, A., & Bardini, R. (2023). Neuronal spike shapes (NSS): a straightforward approach to investigate heterogeneity in neuronal excitability states. *Computers in Biology and Medicine*, 168, 107783.

Mazzoni, A., Panzeri, S., Logothetis, N. K., & Brunel, N. (2008). Encoding of naturalistic stimuli by local field potential spectra in networks of excitatory and inhibitory neurons. *PLOS Computational Biology*, 4(12), e1000239.

McCormick, D. A., Connors, B. W., Lighthall, J. W., & Prince, D. A. (1985). Comparative electrophysiology of pyramidal and sparsely spiny stellate neurons of the neocortex. *Journal of Neurophysiology*, 54(4), 782–806.

Medel, V., Irani, M., Crossley, N., Ossandón, T., & Boncompte, G. (2023). Complexity and 1/f slope jointly reflect brain states. *Scientific Reports*, 13(1).

Merricks, E. M., Smith, E. H., Emerson, R. G., Bateman, L. M., McKhann, G. M., Goodman, R. R., Sheth, S. A., Greger, B., House, P. A., Trevelyan, A. J., & Schevon, C. A. (2021). Neuronal firing and waveform alterations through ictal recruitment in humans. *Journal of Neuroscience*, 41(4), 766–779.

Mosher, C. P., Wei, Y., Kamiński, J., Nandi, A., Mamelak, A. N., Anastassiou, C. A., & Rutishauser, U. (2020). Cellular classes in the human brain revealed in vivo by heartbeat-related modulation of the extracellular action potential waveform. *Cell Reports*, 30(10), 3536–3551.

Moubarak, E., Inglebert, Y., Tell, F., & Goaillard, J.-M. (2022). Morphological determinants of cell-to-cell variations in action potential dynamics in substantia nigra dopaminergic neurons. *Journal of Neuroscience*, 42(40), 7530–7546.

Neher, E., & Sakmann, B. (1976). Single-channel currents recorded from membrane of denervated frog muscle fibres. *Nature*, 260(5554), 799–802.

Orellana, V. D., Donoghue, J. P., & Vargas-Irwin, C. E. (2024). Low frequency independent components: internal neuromarkers linking cortical LFPs to behavior. *iScience*, 27(2), 108310.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Prakash, S. S., Mayo, J. P., & Ray, S. (2022). Decoding of attentional state using local field potentials. *Current Opinion in Neurobiology*, 76, 102589.

Qian, J., & Saggau, P. (1999). Modulation of transmitter release by action potential duration at the hippocampal CA3-CA1 synapse. *Journal of Neurophysiology*, 81(1), 288–298.

Rao, S. G., Williams, G. V., & Goldman-Rakic, P. S. (1999). Isodirectional tuning of adjacent interneurons and pyramidal cells during working memory. *Journal of Neurophysiology*, 81(4), 1903–1916.

Richardson, T. G., Power, G. M., & Smith, G. D. (2022). Adiposity may confound the association between vitamin D and disease risk. *eLife*, 11.

Roshchin, M. V., Matlashov, M. E., Ierusalimsky, V. N., Balaban, P. M., Belousov, V. V., Kemenes, G., Staras, K., & Nikitin, E. S. (2018). A BK channel-mediated feedback pathway links single-synapse activity with action potential sharpening in repetitive firing. *Science Advances*, 4(7).

Shao, L., Halvorsrud, R., Borg-Graham, L. J., & Storm, J. F. (1999). The role of BK-type Ca2+-dependent K+ channels in spike broadening during repetitive firing in rat hippocampal pyramidal cells. *Journal of Physiology*, 521(1), 135–146.

Shu, Y., Hasenstaub, A., Duque, A., Yu, Y., & McCormick, D. A. (2006). Modulation of intracortical synaptic potentials by presynaptic somatic membrane potential. *Nature*, 441(7094), 761–765.

Su, C.-K., Chiang, C.-H., Lee, C.-M., Fan, Y.-P., Ho, C.-M., & Shyu, L.-Y. (2013). Computational solution of spike overlapping using data-based subtraction algorithms. *Frontiers in Computational Neuroscience*, 7.

Trainito, C., von Nicolai, C., Miller, E. K., & Siegel, M. (2019). Extracellular spike waveform dissociates four functionally distinct cell classes in primate cortex. *Current Biology*, 29(18), 2973–2982.

Verkhratsky, A., & Parpura, V. (2014). History of electrophysiology and the patch clamp. *Methods in Molecular Biology*, 1–19.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & SciPy 1.0 Contributors. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272.

Wilson, F. A., O'Scalaidhe, S. P., & Goldman-Rakic, P. S. (1994). Functional synergism between putative gamma-aminobutyrate-containing neurons and pyramidal neurons in prefrontal cortex. *Proceedings of the National Academy of Sciences*, 91(9), 4009–4013.

