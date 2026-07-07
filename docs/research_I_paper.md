**AN INITIAL EVALUATION OF SUPERVISED DENOISING FOR LOW-RAY MONTE CARLO GEOMETRIC-ACOUSTIC IMPULSE RESPONSES**

Jack Rainey

COS 452: Research I

Spring 2026

**Table of Contents**

1. Abstract…………………………………………………………………………………………4

2. Introduction and Problem Statement…………………………...………………………………4

3. Literature Review………………………………………………………………………….……6

3.1. Background……………………………………………………………………...……6

3.2. Related Work..………………………………………………………………………10

4. Methodology and Methods……………………………………………………………………15

4.1. Problem Formulation………………………………………………………………..15

4.2. Paired Dataset Generation…………………………………………………………...16

4.3. Data Representation and Preprocessing……………………………………………..19

4.4. Model Family and Ablation Design…………………………………………………21

4.5. Training Procedure…………………………………………………………………..24

4.6. Evaluation Metrics…………………………………………………………………..26

5. Experiments and Results………………………………………………………………………28

5.1. Experiments and Results Presentation………………………………………………28

5.2. Discussion…………………………………………………………………………...33

6. Future Work…………………………………………………………………………………...35

6.1. Near-Term Model, Dataset, and Evaluation Improvements………………………...35

6.2. Extensions Requiring Simulator or Platform Changes……………………………...38

6.3. Long-Term Extensions Towards General Acoustic Simulation……………………..39

7. Conclusion…………………………………………………………………………………….40

8. Acknowledgements……………………………………………………………………………42

9. Appendix………………………………………………………………………………………42

Appendix A. Simulation Environment and Software Configuration…………………….42

Appendix B. Dataset Configuration, Splits, and Quality Control………………………..43

Appendix C. Learning Representation and Preprocessing Details………………………47

Appendix D. Model Architecture and Ablation Details…………………………………48

Appendix E. Training Configuration and Checkpoint Selection………………………...49

Appendix F. Extended Experiment Discussion………………………………………….54

Appendix G. Asset Attributions………………………………………………………….56

10. Works Cited………………………………………………………………………………….57

**1. ABSTRACT**

Machine learning (ML)-based methods of Monte Carlo simulation denoising have successfully been applied in the graphics domain with production-quality results. However, research regarding ML-based Monte Carlo denoising when applied to the geometric-acoustics domain appears less established. Therefore, this research implements a pipeline whose goal is to denoise ambisonic impulse responses (IRs) using a supervised-learning-based approach by using a low-ray baseline to predict a high-ray reference. The initial model variants created using this approach did not consistently improve over the low-ray baseline under the selected objective metrics, but the experiments identify important limitations in output formulation, path conditioning, and metric design. The research concludes by restating the contributions and encouraging further research into the ML-based geometric-acoustic denoising space.

**2. INTRODUCTION AND PROBLEM STATEMENT**

High-fidelity sound simulation in virtual environments is increasingly important for applications such as virtual production, video games, and virtual reality (VR) [1]. A common approach is geometric acoustics (GA), which simulates sound propagation using Monte Carlo ray-based methods [2], [3]. These methods can produce high-quality spatialized acoustic outputs, but they require large sample counts to reduce Monte Carlo variance, making them computationally expensive [2], [3].

Many spatial-audio workflows represent the effect of an environment using a room impulse response (RIR), assuming that, for fixed geometry and source–receiver positions, sound propagation can be modeled as a linear time-invariant (LTI) system [4]. Under this assumption, the response of the environment to an anechoic (reflection-free) signal is obtained by convolving the input with the RIR [4]. In practice, impulse responses (IRs) can be stored as monaural, stereo, binaural (BRIR), or directionally encoded representations, which preserve spatial characteristics for later decoding or evaluation [5]. From a single impulse response, standardized room acoustic parameters, such as reverberation time or early-to-late energy measures, can be computed using energy decay analysis [4], [5]. However, limited decay range or noise in the impulse response can bias the resulting parameter estimates [5].

The tradeoff between computation and variance has motivated extensive research on denoising Monte Carlo renderings through both algorithmic and machine learning (ML) approaches [6]–[9]. In computer graphics, these methods demonstrate that supervised learning can significantly reduce Monte Carlo noise when low-sample inputs and high-quality reference data are available [7], [9]. However, impulse responses differ substantially from images due to their long temporal structure, extended decay characteristics, and directional encoding, making the direct transfer of image-based denoising methods nontrivial [8], [10], [11]. These differences suggest that acoustic impulse response denoising requires domain-specific modeling.

Motivated by this gap, this work addresses the problem of reducing the computational cost of Monte Carlo geometric-acoustic simulation by denoising low-ray impulse responses. Specifically, this work investigates the following research question: To what extent can machine learning reduce the ray count required for Monte Carlo geometric-acoustic simulation of impulse responses while preserving objective acoustic metrics? This study focuses on ambisonic impulse responses, which preserve directional information while providing a compact representation of spatial sound fields [8].

This research makes three contributions: (1) it formulates ambisonic impulse response denoising for geometric-acoustic simulations as a supervised machine learning problem and identifies key similarities and differences relative to image-based denoising approaches; (2) it implements a data-generation pipeline that produces paired low-ray and high-ray ambisonic impulse responses; and (3) it evaluates whether a learned denoiser can reduce ray count while preserving selected objective acoustic metrics.

**3. LITERATURE REVIEW**

**3.1. Background**

This section briefly reviews the acoustic simulation, spatial audio, and machine-learning concepts used throughout the paper.

**3.1.1 Acoustic Simulation Fundamentals**

Acoustic propagation in an enclosure can be described in terms of sound traveling from a source to a receiver along a direct path and through reflections from surrounding surfaces. In room acoustics, these effects are commonly summarized by a room impulse response (RIR), which characterizes the acoustic transfer between a fixed source and receiver [4]. Under the common assumption that the system is linear and time-invariant (LTI), the response of the environment to an anechoic signal is obtained by convolving the input with the RIR [4]. These properties make the impulse response a central representation for both acoustic simulation and subsequent analysis, and therefore a natural target for denoising.

**3.1.2 Acoustic Simulation Methods**

Acoustic simulation methods are commonly grouped into wave-based, geometric-acoustic, and hybrid approaches [2], [4]. Wave-based methods directly solve acoustic wave equations and capture effects such as interference and diffraction, but they are often computationally expensive for large or complex environments [4]. In contrast, geometric-acoustic methods approximate sound propagation using geometric principles, trading some physical accuracy for improved scalability [2]. Hybrid approaches attempt to combine these methods to balance accuracy and efficiency [4]. Because this work focuses on Monte Carlo geometric-acoustic denoising, the remainder of this section focuses on ray-based GA methods.

Within geometric acoustics, multiple modeling families have been developed over several decades. These methods approximate sound propagation using geometric principles but offer better computational efficiency than wave-based approaches. These methods come at the cost of reduced physical completeness in some regimes.

**3.1.3 Ray-Based Geometric Acoustics**

In ray-based geometric acoustics, sound propagation is approximated by tracing rays emitted from a source through the environment and evaluating how those rays interact with scene surfaces before reaching a receiver [2]. At each surface interaction, part of the sound energy may be reflected while part may be absorbed, with the reflected component continuing along a path determined by the simulation model [2].

Monte Carlo ray tracing estimates these propagation paths through random sampling rather than exhaustive enumeration [13]. More generally, Monte Carlo integration approximates an integral through random sampling, so finite-sample results vary across realizations while converging toward the underlying solution as the number of samples increases [13]. In acoustic simulation, this means that increasing the number of emitted rays improves the stability of the estimated impulse response, while low ray counts introduce variance that manifests as noise in the simulated response [2].

In this context, finite ray counts introduce noise in both the temporal and directional structure of the impulse response, motivating approaches that improve low-ray simulations while preserving the fidelity of higher-ray results [2], [13].

**3.1.4 Ambisonic Impulse Response Representation**

Ambisonics is a three-dimensional spatial-audio representation that describes a sound field using basic functions, typically spherical harmonics, rather than signals tied to a fixed loudspeaker layout [8], [14]. Because the representation is independent of playback configuration, ambisonic signals can be decoded for different loudspeaker arrays or for binaural reproduction [8]. In higher-order ambisonics, increasing the order increases the directional resolution that can be represented [14]. As a result, ambisonic impulse responses provide a compact representation of spatial acoustic information while preserving directional structure relevant to later analysis or rendering [8]. Because ambisonic representations encode both temporal and spatial information, denoising methods must preserve not only signal amplitude over time but also directional consistency across channels [8].

**3.1.5 Characteristics of Acoustic Impulse Responses**

Acoustic room impulse responses (RIRs) are commonly divided into three components: the direct path, early reflections, and late reverberation [15]. The direct path represents line-of-sight propagation, while early reflections arise from a small number of specular interactions with nearby surfaces [15]. Together, these components encode detailed information about the scene’s geometry and surface properties. In contrast, late reverberation consists of dense, diffuse reflections and is more indicative of the environment’s overall size and reverberation characteristics [15].

Directional room impulse responses (DRIRs) extend this representation by preserving spatial information, enabling the reproduction of reverberation effects in three dimensions through multichannel convolution [16]. Because impulse responses span a large dynamic range, limited decay length or noise contamination can bias derived acoustic parameters, particularly those dependent on late reverberation tails [5]. These characteristics make impulse responses challenging to denoise, as methods must preserve both early reflection structure and late reverberation decay without introducing perceptual or metric distortions.

**3.1.6 Neural Networks for Signal Denoising**

Deep learning methods learn task-relevant representations from data through multiple processing layers [10]. A common architecture family is the convolutional neural network (CNN), which applies learned filters across grid-structured inputs [10]. Extensions, such as 3D CNNs, add an additional dimension and are often used when the input exhibits spatial-temporal structure [17]. These architectures are relevant to denoising because they can learn structured temporal and spatial patterns directly from paired noisy and clean data, making them appropriate for supervised impulse response enhancement.

**3.1.7 Objective Room Acoustic Metrics**

Objective room-acoustic evaluation commonly relies on a family of parameters intended to capture different perceptual and physical aspects of a room’s response. Bradley summarizes these parameters in four major groups: decay times, sound strength, clarity measures, and spatial-impression measures, for audience conditions in halls for musical performances [18]. Within this framework, the average reverberation properties of a room are described, while EDT characterizes the initial part of the decay and is often more closely related to perceived reverberance [18]. G represents sound strength, or the effect of the hall on the sound level at the listener's position [18]. Clarity measures, including $D\_{50}$, $C\_{50}$, $C\_{80}$, and $T\_{s}$, are based on the temporal distribution of sound energy over time and are intended to capture different aspects of definition or clarity [18]. Finally, GLL represents spatial measures related to apparent source width and listener envelopment [18]. These parameters are commonly associated with ISO 3382-style room-acoustic evaluation workflows [12], [18]. These metrics are relevant to the present research because the choice of objective evaluation criteria determines which aspects of the denoised impulse responses are treated as successfully preserved.

**3.2. Related Work**

Given the breadth of existing research in related topics, this literature survey focuses only on the work most relevant to the project objectives introduced above. Prior work relevant to this study falls into three broad areas, organized across four subsections: (1) Monte Carlo denoising in computer graphics, (2) geometric-acoustic simulation and spatial audio representations, and (3) machine learning approaches for impulse response modeling. Together, these areas establish the feasibility of learning-based variance reduction and structured acoustic modeling, but do not directly address supervised denoising of Monte Carlo geometric-acoustic impulse responses, particularly in directional representations such as ambisonics.

**3.2.1 Graphics Monte Carlo Denoising**

This work is partially motivated by prior research on denoising Monte Carlo renderings in computer graphics. Before deep learning approaches became common, substantial progress had already been made in algorithmic Monte Carlo variance reduction, but path-traced images at practical sample counts still suffered from visible noise. Kalantari et al. addressed this problem by proposing a machine-learning-based denoising approach trained on example scenes, demonstrating that neural networks could be used effectively to reconstruct cleaner path-traced images from noisy inputs [19]. Bako et al. later addressed limitations in earlier learned filtering approaches by introducing kernel-predicting convolutional networks (KPCN), which use convolutional neural networks to predict spatially varying filters for Monte Carlo denoising [7]. Vogels et al. extended this line of work to animated sequences, where temporal instability introduces additional denoising challenges, by incorporating temporal information, multiscale features, and asymmetric loss functions to improve performance over time [20]. They also showed that the approach generalized beyond a single renderer by evaluating it on RenderMan, Hyperion, and Tungsten [20]. Hou et al. further extended graphics-based Monte Carlo acceleration by proposing a multi-resolution sampling strategy that combines a low-resolution, high-sample rendering with a high-resolution, low-sample rendering to reconstruct a high-resolution, high-sample-quality output using a deep convolutional network [21]. Their formulation is especially relevant to the present study because it shows that supervised reconstruction can exploit complementary information from two differently constrained renderings rather than relying on a single noisy input [21]. Taken together, these studies show that supervised learning can substantially reduce Monte Carlo variance in rendering when paired with low-sample inputs and high-quality reference data are available [7], [19]–[21]. This provides a strong precedent for learning-based variance reduction and suggests that similar ideas may be adapted to acoustic simulation.

**3.2.2 Geometric Acoustics and Spatial IRs**

In acoustics, prior work has primarily focused on simulation rather than post-processing or denoising [3], [11], [22]–[24]. Allen and Berkley addressed the problem of efficiently simulating room impulse responses in enclosed spaces by introducing the image-source method, which models reflections using mirrored virtual sources [22]. Their work became foundational in room-acoustic simulation and helped establish computational modeling of virtual acoustic environments as an important research area [22]. Later, Tsingos et al. addressed a limitation of simpler geometric approaches by incorporating diffraction effects, improving the physical realism of simulated virtual environments while retaining computational efficiency [3]. Funkhouser et al. then addressed the challenge of computing propagation paths for interactive architectural acoustics in large building interiors, proposing a beam-tracing-based approach suited to interactive use [23]. Schissler and Manocha extended this line of work to large, dynamic, multi-source scenes by developing methods for interactive sound propagation and rendering at a greater scale and complexity [24]. At a more applied level, Fırat et al. examined real-time spatial audio rendering in game-engine-based workflows, highlighting the practical importance of interactive acoustic simulation systems for production-oriented environments [11].

In parallel with advances in simulation, spatial audio research has also examined how impulse responses and sound fields should be represented. Merimaa and Pulkki addressed the problem of reproducing spatial room responses across different loudspeaker layouts by introducing spatial impulse response rendering (SIRR), a representation designed to support flexible rendering while preserving natural spatial perception [25]. Pulkki later developed directional audio coding (DirAC) to address limitations in earlier directional rendering approaches and to represent sound fields using spatially structured components rather than purely temporal signals [26]. Together, these works show that impulse responses are not only temporal signals, but also directional representations whose spatial structure is important for accurate rendering and reproduction [25], [26].

Taken together, these studies establish that geometric-acoustic simulation and spatial impulse response representations are mature areas of research [3], [11], [22]–[26]. However, they focus primarily on simulation fidelity and representation rather than variance reduction or denoising of Monte Carlo outputs [3], [11], [22]–[26].

**3.2.3 Simulation Pipelines and Denoising Attempts**

Several recent works have examined impulse responses in the context of practical simulation pipelines, variance reduction, and denoising [16], [27]–[28]. Zang and Kong introduced GSound-SIR, a toolkit for spatial impulse response ray tracing and high-order ambisonic auralization that supports both impulse response synthesis and access to intermediate ray-tracing data [27]. To do so, they extended the GSound ray tracer by decoupling ray generation from the auralization stage, making it possible to reuse traced paths for later analysis or reconstruction [27]. For example, the system allows users to export the top-X or top-X% of rays, which exposes intermediate simulation data that could support learning-based workflows in addition to final impulse-response outputs [27]. The authors also explicitly identify learning-based reconstruction or enhancement from reduced ray sets as a promising direction for future work, further motivating supervised denoising approaches [27].

Related work has also explored non-neural approaches to improving noisy or computationally constrained acoustic simulations [16], [28]. Autio et al. addressed the problem of maintaining acoustic quality in real-time environments, where computational budgets limit the number of rays that can be used in a single pass [28]. They proposed an iterative Monte Carlo acoustic ray-tracing method that progressively refines the simulation by adding samples over time, increasing precision while preserving interactive usability [28]. Massé et al. addressed the problem of non-decaying noise floors in directional room impulse responses, which can distort the late reverberation tail and degrade perceived realism [16]. To address this issue, they used spatially anisotropic reverberation models to resynthesize and prolong the late reverberation tail, thereby reducing noise in the directional response [16].

Together, these studies show that practical acoustic simulation toolchains, variance-aware refinement strategies, and impulse-response denoising are all active areas of research [16], [27]–[28]. However, they do not directly address supervised learning-based denoising of Monte Carlo geometric-acoustic impulse responses using paired low-ray and high-ray data [16], [27]–[28].

**3.2.4 Machine Learning for IRs**

Deep learning has also been applied to several impulse-response modeling tasks adjacent to the present work [29]–[32]. Pezzoli et al. addressed the problem of recovering unknown room impulse responses from partial observations by proposing a CNN-based reconstruction method [29]. Ratnarajah et al. addressed the problem of predicting acoustic impulse responses directly from geometric scene information by introducing MESH2IR, a mesh-based neural model designed for general 3D scenes and real-time neural sound rendering without precomputation [30]. Lin et al. addressed the problem of missing temporal information in room impulse responses by proposing DECOR, a neural model for completing a full response from partial temporal segments [31]. Xia and Zhang addressed the problem of limited spatial resolution in ambisonic impulse responses by using a generative model to upmix lower-order B-format responses to higher-order ambisonic representations [32]. Taken together, these studies show that neural networks can reconstruct, predict, complete, or enhance impulse-response data from incomplete, lower-resolution, or otherwise constrained inputs [29]–[32]. However, they do not directly address denoising Monte Carlo simulation noise in low-ray geometric-acoustic impulse responses, particularly in supervised low-ray/high-ray ambisonic settings [29]–[32].

**3.2.5 Related Work Conclusion**

Taken together, prior work establishes strong precedents for Monte Carlo denoising, acoustic simulation, and neural impulse-response modeling. However, no prior work directly addresses supervised denoising of low-ray Monte Carlo geometric-acoustic impulse responses against high-ray ambisonic references. This gap motivates the present study.

**4. METHODOLOGY AND METHODS**

**4.1. Problem Formulation**

Given this gap, this study formulates Monte Carlo geometric-acoustic impulse response denoising as a supervised learning problem. For a fixed scene configuration—defined by geometry, material properties, source position, receiver position, and simulation parameters—a low-ray ambisonic impulse response and a corresponding high-ray reference are generated using different ray budgets. The objective is to learn a mapping from the low-ray response, optionally augmented with intermediate ray-tracing data, to an estimate of the corresponding high-ray response.

Formally, each training example consists of a tuple $(x\_{low}, z\_{low}, x\_{high})$ where $x\_{low}$ is the low-ray ambisonic impulse response, $z\_{low}$ represents intermediate ray-path features extracted from the same simulation, and $x\_{high}$ is the corresponding high-ray reference generated under identical conditions. A model $f\_{θ}$ is trained to produce an estimate $\hat{x} = f\_{θ}(u)$ that approximates $x\_{high}$. Because each pair is generated from the same underlying acoustic configuration, the primary difference between $x\_{low}$ and $x\_{high}$ is the level of Monte Carlo variance introduced by the ray budget.

This formulation follows the supervised denoising paradigm used in Monte Carlo rendering (Section 3.2.1), where low-sample estimates are mapped to higher-quality references. However, unlike image-based settings, ambisonic impulse responses are long multichannel temporal signals that encode both temporal structure and directional information (Section 3.1.4). As a result, the denoising task requires preserving physically meaningful structure, including direct sound, early reflections, and late reverberation, while maintaining consistency across channels.

This formulation requires evaluation at multiple levels: direct signal reconstruction, preservation of derived room-acoustic metrics, and informal qualitative inspection of rendered preview audio. These evaluation levels are described in Section 4.6.

**4.2. Paired Dataset Generation**

**4.2.1 Simulation Environment and Platform Selection**

At the outset of the project, multiple acoustic simulation environments were considered. The final selection was guided by the need for (1) high-order spatial impulse response generation, (2) access to intermediate ray-tracing data, and (3) compatibility with a reproducible, programmatic pipeline. Based on these criteria, GSound-SIR was selected as the simulation platform.

As discussed in Section 3.2.3, GSound-SIR provides a ray-tracing-based framework for generating spatial impulse responses while exposing intermediate path-level data suitable for learning-based workflows [27]. In particular, its ability to export retained ray paths (e.g., top-k energy paths) enables the construction of auxiliary inputs beyond the final impulse response, which is directly aligned with the supervised formulation introduced in Section 4.1.

To ensure reproducibility, the simulator was vendored into the project and used consistently across all experiments. Minor modifications required for integration are documented in Appendix A. All simulations were executed within a controlled software environment to maintain consistency across dataset generation and model training.

**4.2.2 Procedural Scene Generation**

The dataset was generated procedurally to support scale, diversity, and controlled variation across acoustic conditions. Rather than relying on hand-modeled scenes, a parameterized scene generator was used to sample geometry, material properties, and source–receiver configurations from predefined distributions.

Procedural generation was chosen for three reasons. First, it enables the creation of large paired datasets required for supervised learning. Second, it ensures reproducibility through fixed configuration files and random seeds. Third, it allows controlled distribution shifts to be introduced systematically, enabling evaluation beyond standard in-distribution testing.

The primary dataset consists of indoor acoustic environments generated under a consistent simulation regime. Scenes were divided into multiple splits, including training, validation, in-distribution test, and controlled out-of-distribution test sets that vary material properties, source–receiver placement, and geometry. This design separates generalization performance under nominal conditions from robustness under structured distribution shifts.

Detailed configuration parameters for scene generation, including sampling ranges, split definitions, and simulation settings, are provided in Appendix B.

**4.2.3 Paired Rendering Protocol**

Each scene specification was rendered twice: once using a low-ray budget and once using a high-ray reference budget. Both renders used identical geometry, materials, source and receiver placement, and simulation parameters. The only intended difference between the two was the number of emitted rays.

This paired-rendering strategy follows the supervised data-generation paradigm used in Monte Carlo denoising for rendering (Section 3.2.1), where low-sample estimates are paired with high-quality references. Under this construction, the low-ray impulse response serves as a noisy estimate of the same underlying acoustic configuration represented by the high-ray response.

For each scene, the low-ray render produced both the ambisonic impulse response and the intermediate ray-path data used by path-conditioned model variants. The high-ray render produced the corresponding reference impulse response. Intermediate path data were retained only from the low-ray simulation to reflect the information available in a reduced-computation setting.

All impulse responses were stored in the time domain with consistent sample rate, duration, and ambisonic order, allowing direct comparison between low-ray inputs, high-ray references, and model outputs without additional alignment or resampling.

**4.2.4 Quality Control and Dataset Validation**

After rendering, each example was subjected to a set of quality-control checks before inclusion in the dataset. These checks ensured that impulse responses contained nontrivial energy, that retained path data were present, and that the direct-path arrival occurred at a consistent sample index between the low-ray and high-ray renders within a small tolerance. This requirement ensures that corresponding temporal features represent the same physical events across paired examples. Without this constraint, small shifts in onset time would introduce label noise unrelated to Monte Carlo variance, making the supervised learning problem ill-posed.

Additional constraints were applied to exclude degenerate or incomplete simulations, including cases with negligible impulse-response energy, missing or empty retained path data, or corrupted outputs (see Appendix B for full details). Only examples passing all validation criteria were included in the final dataset, ensuring consistency with the supervised formulation defined in Section 4.1.

**4.3. Data Representation and Preprocessing**

**4.3.1 Time-Domain HOA Learning Representation**

The learning pipeline operated directly on time-domain higher-order ambisonic impulse responses. As discussed in Section 3.1.4, ambisonic impulse responses preserve directional acoustic information, making them suitable for evaluating whether denoising preserves both temporal structure and spatial consistency. In this study, the low-ray HOA impulse response served as the primary model input, while the corresponding high-ray HOA impulse response served as the supervised target.

Stored impulse-response arrays used a channel-first layout, with shape $(C, T)$, where $C$ is the number of ambisonic channels and $T$ is the number of time samples. Before being passed to the neural network, each array was converted to a time-major layout, $(T, C)$, so that the model processed each example as a multichannel temporal sequence. The model, therefore, learned directly from the sampled impulse response rather than from a spectrogram or other time-frequency representation.

This representation was chosen to preserve sample-level correspondence between the low-ray input and high-ray target. Because the paired rendering protocol described in Section 4.2.3 used a consistent sample rate, duration, and ambisonic order, each model output could be interpreted as a full time-domain estimate of the corresponding high-ray HOA impulse response. After inference, predicted responses were converted back to the stored channel-first layout before export and evaluation.

**4.3.2 Path-Feature Representation**

For path-conditioned model variants, the low-ray HOA response was supplemented with retained ray-path information from the same low-ray simulation. These features were included to test whether intermediate simulation data improved denoising beyond the information already present in the impulse response. This design follows the motivation introduced in Section 4.1: the model may use either the low-ray impulse response alone or the low-ray response augmented with intermediate ray-tracing data.

Each path-conditioned example included a fixed-size path-feature matrix constructed from the retained path file associated with the scene. The matrix represented a configured number of retained paths and a fixed set of path-level attributes, allowing path-conditioned examples to be batched consistently across scenes. Identifier-like fields were excluded from the learned feature set so that the auxiliary input represented acoustic propagation information rather than file-order metadata.

The HOA sequence remained the primary acoustic input in these models. The path-feature branch was used only as auxiliary conditioning information, enabling an ablation between HOA-only and path-conditioned variants. Exact retained-path counts, feature definitions, padding behavior, and feature-shape details are reported in Appendix C.

**4.3.3 Normalization and Output Scaling**

Normalization was applied as an internal preprocessing step for model training. HOA inputs and targets were normalized using statistics computed only from the training set. Low-ray input statistics were computed from the training low-ray HOA arrays, while high-ray target statistics were computed from the training high-ray HOA arrays. This separation preserved the distinction between the noisy input distribution and the reference target distribution. The normalization procedure used channel-wise training-set statistics rather than per-example peak normalization. This choice avoided removing meaningful energy differences between scenes, which are relevant to acoustic metrics and preview-audio rendering. Path features, when used, were normalized separately using feature-wise training-set statistics after fixed-size feature matrices were constructed.

During inference and export, predicted outputs were denormalized using the target HOA statistics and saved in the same time-domain HOA format as the original simulation outputs. Consequently, signal-level comparisons, objective room-acoustic metric evaluation, qualitative inspection, and preview-audio rendering were all performed on denormalized impulse responses rather than on normalized network outputs.

**4.3.4 Preview-Audio Rendering**

Although the models were trained on impulse responses rather than ordinary audio clips, perceptual inspection required human-interpretable audio examples. For this purpose, selected low-ray, high-ray, and denoised HOA impulse responses were rendered into preview audio by applying them to short anechoic source signals. These preview files were treated strictly as evaluation artifacts. They were not used as model inputs or targets. The canonical data representation throughout training and objective evaluation remained the denormalized time-domain HOA impulse response.

**4.4. Model Family and Ablation Design**

**4.4.1 Temporal CNN Architecture**

This study used a one-dimensional convolutional neural network as the primary model family for HOA impulse-response denoising. This choice was motivated by the structure of the learning problem: each example is a long, multichannel temporal sequence, and the model must produce an output with the same sample length and channel count as the high-ray target. As discussed in Section 3.2.1, convolutional architectures have been used effectively in supervised Monte Carlo denoising, but the acoustic setting requires the convolutional structure to operate over time rather than over image space.

The base model, therefore, applied temporal convolutions to normalized HOA sequences in time-major form. The network used the same-padding convolutional layers so that the output remained sample-aligned with the input. This was important because the paired-rendering protocol described in Section 4.2.3 assumes that corresponding low-ray and high-ray samples represent the same underlying acoustic event. The model was not intended to predict a compressed representation, acoustic metric vector, or perceptual score; instead, it predicted a full-time-domain HOA impulse response suitable for the same downstream evaluation pipeline as the simulated references.

Residual connections were used within the convolutional stack to improve optimization stability and to support correction of an already informative low-ray input. This design reflects the assumption that the low-ray response contains substantial acoustic structure, even though it also contains Monte Carlo variance. Full architectural specifications, including kernel sizes, dilation schedule, layer widths, and parameter counts, are provided in Appendix D.

**4.4.2 Output-Target Ablation: Full Prediction vs. Residual Prediction**

Two output formulations were evaluated: full prediction and residual prediction. In the full-prediction formulation, the network directly predicts the normalized high-ray HOA response from the normalized low-ray input. This formulation gives the model freedom to reconstruct the target response without explicitly preserving the low-ray input.

In the residual-prediction formulation, the network predicts a correction term that is added to the normalized low-ray input. This formulation biases the model toward preserving the structure already present in the low-ray simulation while learning the difference between the low-ray estimate and the high-ray reference. The residual formulation is therefore especially natural for this task because the low-ray and high-ray renders correspond to the same scene configuration and differ primarily in Monte Carlo variance.

This ablation tested whether low-ray acoustic denoising is better treated as direct reconstruction or as correction of a noisy but physically meaningful simulation. Comparing these formulations also helped identify whether the model benefited from a stronger architectural bias toward preserving the input response.

**4.4.3 Input Ablation: HOA-Only vs. Path-Conditioned Models**

Two input configurations were evaluated. The HOA-only model received only the normalized low-ray HOA impulse response. This configuration tested whether the temporal structure of the low-ray impulse response alone contained sufficient information for denoising.

The path-conditioned model received both the normalized low-ray HOA sequence and a fixed-size matrix of retained path features from the corresponding low-ray simulation. This configuration tested whether intermediate ray-tracing information improved denoising beyond the impulse response itself. The auxiliary path branch encoded the retained path matrix into a learned embedding, which was then combined with the temporal HOA representation before prediction.

This ablation directly follows the problem formulation in Section 4.1, where the model input may consist of the low-ray response alone or the low-ray response augmented with intermediate simulation data. By comparing HOA-only and path-conditioned variants while keeping the main temporal CNN structure consistent, the study isolated the contribution of retained ray-path information to denoising performance. Detailed path-feature dimensions and branch architecture are reported in Appendix D.

**4.5. Training Procedure**

All model variants were trained under the same supervised learning framework described in Section 4.1. Each training example paired a low-ray HOA impulse response, optionally supplemented with low-ray path features, with its corresponding high-ray HOA reference. The training procedure was designed to compare model variants under consistent conditions rather than to independently tune each architecture as a separate system.

**4.5.1 Training Objective**

Models were trained using a tail-aware impulse-response reconstruction objective. The primary waveform term was based on the Huber loss, which applies a quadratic penalty to small residuals and a linear penalty to larger residuals. This loss was chosen because impulse responses contain both low-amplitude reverberant structure and large localized events, such as direct sound and early reflections. Compared with a purely squared-error objective, the Huber loss reduces the influence of large localized residuals; compared with a purely absolute-error objective, it remains sensitive to small reconstruction errors. This follows Huber’s robust-estimation formulation, which was designed to provide an intermediate behavior between mean-like and median-like estimation under non-ideal error distributions [33].

The loss was computed after denormalizing the predicted and target HOA responses back into raw amplitude space. This ensured that the training objective reflected physically meaningful signal amplitudes rather than only normalized network units. In addition to the waveform reconstruction term, the training objective included late-window and energy-decay components to emphasize preservation of the reverberant tail. These additions were included because late reverberation is central to room-acoustic evaluation, while the early direct sound and reflection structure can otherwise dominate sample-wise waveform losses. Exact loss weights and training hyperparameters are reported in Appendix E.

For full-prediction models, the loss was applied between the predicted HOA response and the high-ray target. For residual-prediction models, the predicted residual was first added to the low-ray input, and the resulting estimate was compared against the same high-ray target. Thus, all model variants were optimized against the same high-ray reference representation, differing only in input conditioning and output parameterization.

**4.5.2 Optimization Protocol**

Each model variant was trained using the same optimization protocol to ensure that comparisons reflected differences in model formulation rather than differences in training procedure. The training split was used for parameter updates, and the validation split was used to monitor generalization during training. No test split was used for optimization, hyperparameter selection, or checkpoint selection. The optimizer, learning-rate behavior, batch size, epoch limits, early-stopping settings, and checkpointing procedure were held fixed across the primary ablation runs. These implementation-level parameters are reported in Appendix E.

**4.5.3 Model Selection and Validation**

For each training run, model selection was based on validation-set behavior rather than test-set performance. The checkpoint retained for evaluation was selected according to the monitored validation objective, so the test sets remained reserved for final comparison only. This separation was important because the study included both in-distribution and controlled out-of-distribution test splits. Using any test split during model selection would have weakened the intended distinction between validation-based development and held-out evaluation. Validation performance was used primarily to identify the best checkpoint within a run and to detect overfitting or unstable training behavior. Final conclusions were based on the evaluation procedure described in Section 4.6, where the selected checkpoint was applied to held-out low-ray examples and compared against the corresponding high-ray references.

**4.5.4 Training and Export Pipeline**

Training was implemented as part of the same pipeline used for preprocessing, inference, export, and evaluation. During training, inputs were loaded in the normalized time-major representation described in Section 4.3. For path-conditioned models, normalized path-feature matrices were loaded alongside the HOA inputs. After inference, predictions were denormalized and exported through the procedure described in Appendix C.4. Exact configuration files, checkpoint paths, and training parameters are provided in Appendix E.

**4.5.5 Relationship to Later Experiments**

The training procedure supported the ablation structure introduced in Section 4.4. Because all variants shared the same training, validation, export, and evaluation pipeline, later experiments could compare input conditioning and output formulation under a consistent methodological framework. Final comparisons were based on the evaluation procedure described in Section 4.6 rather than on training or validation loss alone.

**4.6. Evaluation Metrics**

Model outputs were evaluated by comparing low-ray inputs and denoised predictions against their corresponding high-ray references. All objective evaluation was performed after inference and export using denormalized time-domain HOA impulse responses.

**4.6.1 Signal-Level Evaluation**

Signal-level evaluation measured direct reconstruction accuracy between each evaluated response and the high-ray reference. For each scene, the raw low-ray response served as the baseline condition, and the denoised model output was evaluated against the same high-ray target. Metrics included mean squared error, mean absolute error, root mean squared error, relative L2 error, peak absolute error, and signal-to-noise ratio relative to the high-ray reference. Improvement ratios were computed by comparing the low-ray error to the denoised-output error, allowing each model output to be interpreted relative to the reduced-ray input it was intended to improve. Relative L2 error was computed as the L2 norm of the difference between an evaluated response and the high-ray reference, normalized by the L2 norm of the high-ray reference. This follows the common use of relative norm error in reconstruction tasks, where the reconstruction error is scaled by the reference signal magnitude. The signal noise ratio (SNR) was computed as the ratio of reference signal power to reconstruction-error power and reported in decibels, following the standard definition of SNR as signal power divided by noise power [34].

**4.6.2 Room-Acoustic Metric Evaluation**

Room-acoustic metric evaluation measured whether denoising preserved selected scalar acoustic parameters derived from the impulse response. For this initial evaluation, metrics were computed from the first HOA channel, which was treated as the omnidirectional component of the response for scalar metric computation. The evaluated metrics included EDT, $T\_{20}$, $T\_{30}$, $C\_{50}$, $C\_{80}$, $D\_{50}$, and center time. Each metric was computed for the low-ray input, denoised output, and high-ray reference, and errors were reported relative to the high-ray condition.

Spatial room-acoustic metrics were not included in the initial objective evaluation because they require an additional HOA decoding or spatial-analysis convention beyond the scalar impulse-response metrics implemented here.

**4.6.3 Preview Audio Evaluation**

Preview-audio evaluation was used for qualitative listening inspection rather than quantitative model selection. The low-ray, denoised, and high-ray preview renders were compared to identify audible artifacts that might not be fully captured by signal-level or scalar room-acoustic metrics. These previews were not used for training, checkpoint selection, or objective metric computation. Because no controlled listening study was conducted, these previews were used only for informal inspection and should not be interpreted as evidence of perceptual preference or perceptual equivalence.

**5. EXPERIMENTS AND RESULTS**

**5.1. Experiments and Results Presentation**

**5.1.1 Experimental Overview**

Given the methodology described above, the experiments evaluated four model variants formed by crossing two output formulations, full-response prediction and residual prediction, with two input configurations, HOA-only input and HOA-plus-path conditioning. All variants were trained and evaluated under the shared pipeline described in Sections 4.5 and 4.6, allowing differences in performance to be interpreted primarily as effects of output formulation and input conditioning rather than as changes in data processing, training procedure, or evaluation criteria. For all experiments, model success was defined relative to the low-ray baseline. A denoised output was considered successful when it was closer to the high-ray reference than the corresponding low-ray input was, meaning that error-based metrics decreased relative to the baseline and SNR increased relative to the baseline. This criterion reflects the purpose of the denoiser: to make a reduced-ray simulation measurably more similar to the high-ray reference, not merely to produce a plausible impulse response.

**5.1.2 Low-Ray Baseline Error Across Splits**

As defined in Appendix B.1, the low-ray input contained $5,000$ rays, while the high-ray reference contained $200,000$ rays for all experiment scenes. Because each pair used the same scene configuration and differed only in ray budget, the low-ray response served as the reduced-computation baseline against which later denoised outputs were compared.

*Figure 1: Low-ray baseline L2 and SNR comparisons to high-ray reference by split*

| **Split** | **Low rel. L2** | **Low SNR** |
| --- | --- | --- |
| test\_id | $0.504$ | $6.306$ |
| test\_material\_shift | $0.4918$ | $6.354$ |
| test\_placement\_shift | $0.502$ | $6.355$ |
| test\_geometry\_shift | $0.6081$ | $4.628$ |

The baseline metrics show that the low-ray input differed measurably from the high-ray reference while still remaining a meaningful approximation of the same acoustic scene. The in-distribution, material-shift, and placement-shift splits had similar baseline error, while the geometry-shift split showed a larger relative L2 error and lower SNR. This indicates that the $5,000$-ray input provided an informative reduced-computation simulation, but still left measurable signal-level error for the denoising models to improve.

**5.1.3 Model Variant Comparison**

*Figure 2: Prediction quality for the four models on the test\_id split*

| **Run** | **Low rel. L2** | **Pred rel. L2** | **Rel. L2 ratio** | **ΔSNR (dB)** |
| --- | --- | --- | --- | --- |
| hoa\_direct | $0.504$ | $7.417$ | $0.0684$ | $-23.51$ |
| hoa\_residual | $0.504$ | $0.518$ | $0.9717$ | $-0.2494$ |
| path\_direct | $0.504$ | $1.304$ | $0.3867$ | $-8.595$ |
| path\_residual | $0.504$ | $0.5134$ | $0.9808$ | $-0.1684$ |

On the in-distribution test split, none of the four model variants satisfied the success criterion defined in Section 5.1.1. The full-prediction models performed substantially worse than the low-ray baseline, indicating that direct reconstruction of the high-ray response was unstable or poorly suited to the tested architecture and training configuration. The residual-prediction models were far more stable and remained close to the low-ray baseline, but they still failed to improve relative L2 error or SNR. Thus, residual prediction appeared to preserve the structure of the reduced-ray input more effectively than full prediction, but it did not produce a measurable improvement over that input. Path conditioning improved the full-prediction model relative to the HOA-only full-prediction model, but this improvement was insufficient to make direct prediction competitive with the low-ray baseline.

Additional discussion regarding model variants can be found in Appendix F.

**5.1.4 Generalization Across Shifted Test Splits**

*Figure 3: Prediction quality for the path\_residual model by testing splits*

| **Split** | **Low rel. L2** | **Pred rel. L2** | **Rel. L2 ratio** | **ΔSNR** |
| --- | --- | --- | --- | --- |
| test\_id | $0.504$ | $0.5134$ | $0.9808$ | $-0.1684$ |
| test\_material\_shift | $0.4918$ | $0.5018$ | $0.9814$ | $-0.1639$ |
| test\_placement\_shift | $0.502$ | $0.5126$ | $0.9785$ | $-0.1888$ |
| test\_geometry\_shift | $0.6081$ | $0.6218$ | $0.9787$ | $-0.1871$ |

This subsection focuses on the path\_residual model variant because it produced the highest relative L2 ratio among the four tested models on the in-distribution test split. Additional discussion regarding the other model variants can be found in Appendix F.

Across the four test splits, the path\_residual model performed similarly relative to the low-ray baseline. The relative L2 ratio remained slightly below 1.0 for every split, and ΔSNR remained negative in every case. Therefore, as in Section 5.1.3, the model failed to provide a meaningful signal-level improvement over the low-ray input. However, the shifted splits did not show a substantially different failure pattern from the in-distribution split. Even though the geometry-shift split had the highest baseline error, the model’s relative degradation remained similar to the other splits.

**5.1.5 Room-Acoustic Metric Preservation**

*Figure 4: Acoustic metrics for the path\_residual model on the test\_id split*

| **Metric** | **Low**  **abs. error** | **Pred abs. error** | **Error reduction** | **Fraction improved** | **Median reduction** |
| --- | --- | --- | --- | --- | --- |
| EDT (s) | $0.1095$ | $0.1102$ | $-6.6992×10^{-4}$ | $23.3\%$ | $-9.0722×10^{-4}$ |
| T20 (s) | $0.0613$ | $0.05371$ | $0.007588$ | $85.0\%$ | $0.006006$ |
| T30 (s) | $0.0385$ | $0.02863$ | $0.009869$ | $90.0\%$ | $0.01013$ |
| C50 (dB) | $1.883$ | $1.908$ | $-0.0243$ | $15.0\%$ | $-0.01415$ |
| C80 (dB) | $1.356$ | $1.413$ | $-0.0567$ | $18.3\%$ | $-0.03717$ |
| D50 (%) | $6.66$ | $6.7$ | $-0.04015$ | $15.0\%$ | $-0.04941$ |
| Center time (s) | $0.004183$ | $0.00421$ | $-2.695×10^{-5}$ | $20.0\%$ | $-6.2522×10^{-5}$ |

For the same reasons as Sections 5.1.3 and 5.1.4, this section focuses on the path\_residual model on the in-distribution test split. Additional discussion of the other model variants is provided in Appendix F.

The scalar room-acoustic metrics showed a more mixed pattern than the signal-level metrics. Although the path\_residual model failed to improve relative L2 error or ΔSNR, it reduced error for the T20 and T30 decay-time metrics on the in-distribution test split. However, this improvement did not extend across the full set of evaluated acoustic metrics. EDT, C50, C80, D50, and center time either changed only slightly or moved farther from the high-ray reference. These results suggest that waveform-level similarity and scalar room-acoustic metric preservation are related but not interchangeable evaluation targets. In particular, the model may preserve or slightly improve some decay-slope-derived quantities without producing a better sample-level reconstruction of the high-ray impulse response.

Therefore, later discussion treats objective metric preservation as metric-dependent rather than as a single uniform outcome.

**5.1.6 Qualitative Preview-Audio Inspection**

Qualitative inspection was performed using the preview-audio procedure described in Section 4.6.3. Because no controlled listening study was conducted, these inspections were used only informally and should not be interpreted as evidence of perceptual preference or perceptual equivalence. In this informal inspection, no obvious audible differences were identified among the $5,000$-ray input, the $200,000$-ray reference, and the predicted output. Although this observation cannot support a formal perceptual conclusion, it suggests that the low-ray baseline may already have been perceptually plausible for the inspected examples. This provides one possible interpretation of the mixed results in Sections 5.1.2–5.1.5: the model may have been asked to improve an input that was already close to the reference in ways not easily detected through informal listening.

**5.1.7 Summary of Findings**

Across the evaluated model variants and test splits, the experiments produced mixed but informative results. Full-prediction models performed poorly at signal-level reconstruction, while residual-prediction models remained much closer to the low-ray baseline but still failed to produce meaningful relative L2 or SNR improvement. The selected residual model behaved similarly across the in-distribution and shifted test splits, suggesting that its failure mode was stable rather than specific to one test condition. Room-acoustic metrics showed a more selective pattern: some decay-time metrics improved, but metric preservation was not consistent across the full evaluated metric set. Informal preview-audio inspection also suggested that the low-ray baseline was already perceptually plausible for the inspected examples, though this observation was not part of a controlled listening study. Overall, the tested models did not meet the study’s success criterion, but the results clarify several limitations that should guide future model design and evaluation.

**5.2. Discussion**

The experiments show that the proposed pipeline was able to generate paired low-ray and high-ray ambisonic impulse responses, train supervised model variants, export predictions, and evaluate those predictions against high-ray references. However, the tested models did not meet the success criterion defined in Section 5.1.1. None of the evaluated variants improved over the low-ray baseline according to the primary signal-level reconstruction metrics. Therefore, the results should be interpreted as an initial diagnostic evaluation rather than as evidence that the tested formulation successfully reduced the ray count required for geometric-acoustic impulse-response simulation.

Several factors may explain this limited performance. The tested temporal CNN architecture may not have been well matched to the structure of ambisonic impulse responses, which contain sparse early reflections, long reverberant tails, and directional information across channels. The training configuration may also have required broader hyperparameter tuning, especially because earlier development suggested that some final-model behavior may have depended on optimization or loss-weighting choices. The difference between full-prediction and residual-prediction behavior also suggests that output formulation remains an unresolved design choice rather than a settled modeling conclusion. In addition, the $5,000$-ray input may have been too strong a baseline, leaving limited room for measurable improvement under the selected metrics and informal preview-audio inspection. Because these possibilities were not isolated experimentally, the current results do not identify a single cause for the limited model performance.

The results also suggest that evaluation for this task cannot rely on only one type of metric. Signal-level metrics showed that the denoised outputs did not improve relative L2 error or SNR, but selected room-acoustic metrics showed a more mixed pattern, with some decay-time metrics improving while clarity, definition, and center-time metrics did not improve consistently. This indicates that waveform reconstruction and acoustic-parameter preservation are related but not interchangeable goals. Future versions of the training and evaluation pipeline may therefore need to target room-acoustic behavior more directly, rather than assuming that lower waveform error will automatically produce better acoustic metrics.

The study’s conclusions are also limited by the scope of the dataset and evaluation. The experiments used procedurally generated indoor scenes, static source–receiver configurations, and a limited set of material, placement, and geometry shifts. Scalar room-acoustic metrics were computed from the first HOA channel, so the evaluation did not fully assess spatial preservation across the ambisonic representation. In addition, the preview-audio inspection was informal and should not be treated as a controlled perceptual study. These limitations suggest that the present work should be viewed as a first-pass evaluation of supervised denoising for Monte Carlo geometric-acoustic impulse responses rather than as a general conclusion about the feasibility of learning-based variance reduction in geometric acoustics.

**6. FUTURE WORK**

Given the limitations defined above, the research motivates several directions for future work, ranging from near-term small iterations to long-term speculative categories of research.

**6.1. Near-Term Model, Dataset, and Evaluation Improvements**

Most importantly, further research should investigate the reasons for the limited model performance and the overall feasibility of using deep learning to improve low-ray geometric-acoustic inputs. Preferably, this research would result in the creation of models meeting the definition of success in Section 5.1.1, but further research confirming the unsuitability of deep-learning models for the given task is also possible. In addition, these models should ideally improve both training and evaluation metrics. Further research may also be required to explain why the models performed better on some metrics than on others. Either way, the current stage of research is unable to conclude definitively the performance of deep learning models for this task.

To accomplish this, a variety of tasks are recommended in future work. To begin, the current pipeline, to our knowledge, works correctly and reproducibly. However, code readability and usability became increasingly difficult as project scope increased. Therefore, major rewrites of the code base are recommended to improve future scalability of experimentation. In addition, the pipeline was written to support the current stage of research, and additional edits may be required to carry out many of the tasks in the remainder of the future work section. Not only that, the improvements to the pipeline hope to eliminate the need for GSound-SIR modification to the given vendor of the code for research. Also, the current pipeline lacks powerful data visualization capability, meaning that research explanation is difficult to explain in a visually enjoyable manner. These visualization capabilities could range from basic automated graph rendering up towards highly complex game-engine and/or graphics render-engine-based simulations.

Many of the recommended experiments involve further research into parameters that currently remain fixed. First, vary the low-ray input count from the currently constant $5,000$ ray input to evaluate model performance across different levels of input. Next, perform a more in-depth hyperparameter search to improve the model’s ability to converge on a useful solution. In addition, consider additional deep learning models in addition to the CNN architecture defined in Appendix D to test the overall performance of the chosen model as well as research the success of other deep-learning model paradigms. In addition, further research could also adapt the multi-resolution sampling idea discussed in Section 3.2.1 by combining low-ray, high-order ambisonic inputs with high-ray, low-order ambisonic inputs to predict a high-ray, high-order ambisonic reference [21]. Given the additional research of model paradigms, it may also be beneficial to do a brief exploration of the complexity cost of any given model related to its overall accuracy on a variety of metrics.

A completely controlled human perception survey would also be a valuable area of research because while improvement on objective metrics is valuable, if the model fails to improve human perception, or actively makes the output less preferable, the model fails a significant category of testing. Therefore, performing a controlled perception survey would improve the overall completeness of this research. In addition, this means that further pipeline improvement to the qualitative previews will be required to ensure that playback can occur without compromising spatial output and output quality. It is also worth noting that while addressing the limitations required for a human perception survey, the objective metric limitations discussed in Section 4.6.2 should also be accounted for in this pipeline upgrade.

Next, consider further research into why the model performs better for specific inputs than others, such as why the model discussed in Section 5.1.4 performed slightly better on the test\_material\_shift split than the general test\_id split. In addition, the current dataset only uses a limited list of materials and room geometries. Extending this list would most likely improve model generalization and research completeness. Not only that, the current research only considered fully-enclosed room impulse response environments. Extending the dataset to partially open and fully outdoor environments would further test the generality of the simulation system.

Future work should also include a more explicit statistical analysis plan for interpreting model improvements. Because several splits in the present study used relatively small sample sizes, later experiments should estimate the minimum detectable effect size or expected confidence-interval width for the primary evaluation metrics before drawing strong conclusions from small differences between the low-ray baseline and denoised outputs. This would help distinguish practically meaningful improvement from variation caused by limited sample size. In addition, future work should include a structured failure-case analysis. For each model variant, poorly performing scenes could be grouped by scene geometry, material distribution, source–receiver placement, baseline low-ray error, reverberation characteristics, and retained-path statistics. Such analysis would help determine whether denoising failures are associated with identifiable acoustic or geometric conditions rather than treating model performance as a single aggregate result.

**6.2. Extensions Requiring Simulator or Platform Changes**

All of the future work mentioned thus far can be implemented using the existing GSound-SIR implementation. However, the vendor of GSound-SIR used contains multiple important limitations that would require either significant changes to the GSound-SIR codebase or choosing a new geometric-acoustic raytracer to solve future work problems. It is also possible that these changes would require writing a custom raytracer to accomplish these given research goals.

To begin, the vendor of GSound-SIR only supported static scenes and a single-source receiver position. As a result, research regarding model generalization with time dimensionality could not be studied. In addition, the practicality of the model, as multiple source and receiver positions are introduced, could not be studied. Being able to see this research would improve research completeness.

In addition, while not essential to core research, being able to run the pipeline on cross-platform environments would improve overall research reproducibility. Most notably, a test of the pipeline in the macOS environment resulted in an invalid processor error that most likely indicates that the Apple Silicon/ARM platform is incompatible with GSound-SIR. Getting the pipeline working in this environment was left to future work.

**6.3. Long-Term Extensions Towards General Acoustic Simulation**

The preceding sections describe near-term work needed to complete and extend the present denoising pipeline. Beyond those extensions, several longer-term directions would move the project from an initial low-ray/high-ray denoising study toward a broader research program in learned acoustic simulation. These directions would require substantial additional literature review and implementation work, rather than as direct continuations of the current research.

For one, being able to generalize the pipeline to use multiple raytracers instead of only GSound-SIR would allow future work to test whether learned denoising models capture simulator-independent acoustic structure or instead overfit to artifacts of a particular ray-tracing implementation. In addition, being able to compare multiple different acoustic simulation systems, not only geometric acoustic Monte Carlo-based rendering systems, would allow for research regarding denoising across different simulation paradigms. This would allow for research regarding denoising in hybrid simulation systems and comparison of the overall computation cost of different simulation paradigms, not only from the system’s raw output, but from a deep-learning denoised output.

Savioja and Svensson also noted a gap between the active investigation into accurate room acoustic modeling compared to that of accurate room graphics modeling for real-time or near real-time environments in the entertainment industry [2]. Contingent upon the models meeting the definition of success in Section 5.1.1, ML-based geometric acoustic denoising may help close this research gap and improve acoustic raytracing under these conditions.

Another avenue for future work focuses on the physical completeness of the simulation environment to represent key environmental changes. For example, a researcher may wish to compare the simulation of a sound made on the surface of Earth to that of the surface of Mars. Due to the separate properties of each of these environments, building into the raytracer the ability to account for these environments may encourage further usage and research in the domain. This most likely would involve being able to modify sound absorption rates and similar parameters in scientifically controlled manners.

A further long-term direction is the development of user-facing acoustic design tools built around existing 3D content-creation environments. Blender is a free and open-source 3D creation suite, making it a plausible front-end environment for experimental acoustic authoring tools [35]. In such a system, Blender would not necessarily perform the acoustic simulation itself; rather, it could provide the scene-authoring interface through which users define geometry, materials, source objects, receiver positions, and listener paths. An acoustic backend could then generate spatial impulse responses for selected source–receiver configurations and return preview audio to the user. For example, a concert-hall designer could place loudspeakers or instruments in a modeled room, move a virtual listener through different seating locations, and compare predicted acoustic conditions under alternative material or occupancy assumptions. Ideally, the system would support fast approximate previews during design iteration and higher-quality offline renders for final evaluation. This direction would extend the present work from offline denoising of simulated impulse responses toward interactive acoustic design workflows.

**7. CONCLUSION**

This research evaluated the following research question: to what extent can machine learning reduce the ray count required for Monte Carlo geometric-acoustic simulation of impulse responses while preserving objective acoustic metrics? To answer this question, the study made three contributions: (1) it formulated ambisonic impulse response denoising for geometric-acoustic simulations as a supervised machine learning problem and identified key similarities and differences relative to image-based denoising approaches; (2) it implemented a data-generation pipeline that produces paired low-ray and high-ray ambisonic impulse responses; and (3) it evaluated whether a learned denoiser can reduce ray count while preserving selected objective acoustic metrics.

The research began with a literature survey situating the project within Monte Carlo rendering denoising, geometric-acoustic simulation, spatial impulse-response representation, and machine-learning-based impulse-response modeling. This review found that machine-learning-based Monte Carlo denoising is well established in image rendering, but that it appears less established in geometric acoustics. At the same time, extensive research has addressed geometric-acoustic simulation and algorithmic denoising. In addition, machine learning has also been meaningfully applied in the geometric-acoustic space, but not for the specific task of geometric-acoustic simulation denoising, despite being identified as a potential avenue for future research [27].

Given the identified research gap, this work considered geometric-acoustic Monte Carlo denoising as a supervised learning problem. To acquire the data necessary to train a supervised learning model, the GSound-SIR ray tracer was used to generate a procedural dataset meeting the project parameters [27]. This dataset was used to train a CNN model, which was chosen due to the temporal structure of the impulse-response data. Multiple model variants were also compared, including full HOA prediction versus residual prediction and HOA-only input versus HOA input augmented with intermediate ray-path data. In experiments, the models produced generally negative but mixed results, improving some scalar acoustic metrics while failing to improve the primary signal-level metrics and several other acoustic measures. These results also indicate that waveform reconstruction and acoustic-parameter preservation are related but not interchangeable goals.

At the present stage, the results do not determine whether the limited performance reflects a fundamental mismatch between the tested methodology and the task or limitations of the current dataset, architecture, loss design, and hyperparameter settings. Therefore, future work should address both near-term improvements to the dataset, model design, and evaluation pipeline and longer-term extensions toward more general acoustic simulation systems. Although the tested models did not demonstrate successful ray-count reduction, this work establishes a reproducible experimental foundation for determining whether supervised learning can become a practical variance-reduction tool for geometric-acoustic simulation.

**8. ACKNOWLEDGEMENTS**

Special thanks to Dr. Fola Ayano for research mentorship and to Dr. Jon Denning for initial project formulation and direction. I also thank Zang and Kong for developing the GSound-SIR library that made the present research possible, as well as the authors who provided the assets used for the listening previews. These assets are listed in Appendix G. I also acknowledge OpenAI’s ChatGPT for assistance in compiling research sources, writing portions of the research code, and refining wording. Finally, I acknowledge Grammarly and MyBib for assistance with proofreading, formatting, and bibliography management.

**9. APPENDIX**

**Appendix A. Simulation Environment and Software Configuration**

GSound-SIR was used as the acoustic simulation platform for dataset generation because it supports spatial impulse response ray tracing, high-order ambisonic auralization, and export of intermediate ray-path data. The simulator was downloaded from GitHub on March 19, 2026, using the then-current latest commit, and was vendored into the project repository to ensure that all experiments used a fixed simulator implementation.

One minor source modification was made to support integration with the project pipeline. In auralizer/setup.py, line 67, the identifier spherical\_harmonics was changed to spherical\_harmonics\_rt. No other simulator-level changes were made for the experiments reported in this study.

All experiments were conducted on an Ubuntu 24.04.3 LTS desktop system equipped with an Intel® Core™ i7-14700K x 28 CPU, an NVIDIA T1000 8 GB GPU, Intel UHD Graphics 770, and 32.0 GiB of RAM. The software environment included C++ 13.3.0, Python 3.12.3, SoundFile 0.13.1, TensorFlow 2.21.0, and wheel 0.46.3. This environment was used for simulation, preprocessing, model training, inference, and evaluation.

**Appendix B. Dataset Configuration, Splits, and Quality Control**

**B.1 Primary Simulation Configuration**

The primary dataset was generated using a fixed simulation configuration. All impulse responses were rendered at a $48$ kHz sample rate with a duration of 3.0 s. Third-order ambisonics was used, producing 16 channels per impulse response according to $(N+1)^{2}$ , where $N=3$. The low-ray budget was $5,000$ rays, and the high-ray reference budget was $200,000$ rays. Intermediate path data were retained from the low-ray simulation using a top-$k$-energy policy with $k=5,000$. Stored impulse-response arrays used float32 precision.

**B.2 Dataset Splits**

The final dataset was divided into six splits: training, validation, in-distribution test, material-shift test, placement-shift test, and geometry-shift test. The training, validation, and in-distribution test splits shared the same underlying scene-generation distribution and differed primarily by random seed. The material-shift split altered the material distribution while preserving the primary geometry and placement regime. The placement-shift split preserved the primary geometry and material distribution but biased source–receiver placement toward more difficult near-corner cases. The geometry-shift split used corridor scenes rather than shoebox rooms, introducing a controlled geometry change.

These splits were designed to separate ordinary held-out performance from robustness under structured distribution shifts. The in-distribution test split evaluates whether the model generalizes to unseen examples drawn from the same procedural distribution as training, while the shifted splits evaluate whether the model remains useful when materials, placement, or geometry differ from the training distribution.

**B.3 Scene Generation Scope**

Although the procedural generator supported multiple indoor geometry families, the primary in-distribution data used shoebox rooms with the interior\_random+mid\_pair placement regime. This study was limited to indoor room impulse responses in order to reduce uncontrolled variation and focus the initial investigation on Monte Carlo variance in enclosed acoustic scenes.

**B.4 Quality-Control Criteria**

Each rendered example was accepted only if it passed the dataset quality-control checks. These checks required nontrivial impulse-response energy, non-empty retained path data, retained-path files within the configured size bound, and a direct-path onset mismatch within the configured tolerance between the low-ray and high-ray renders. The onset check ensured that the same physical arrival was represented at a consistent sample location across each paired example. Examples failing these checks were excluded from the dataset. This prevented degenerate renders, incomplete path exports, corrupted outputs, or poorly paired responses from entering the supervised training set.

**B.5 Versioned Configuration Files**

Dataset generation was controlled by versioned JSON configuration files rather than by a purely command-line interface. These files specified split sizes, random seeds, geometry sampling ranges, material regimes, source–receiver placement constraints, simulation parameters, retained-path settings, and quality-control thresholds.

*Figure 5: Primary dataset configuration summary*

| **Configuration category** | **Value** |
| --- | --- |
| Base seed | $42$ |
| Sample rate | $48,000$ Hz |
| IR duration | $3.0$ s |
| Ambisonic order | $3$ |
| Low-ray budget | $5,000$ rays |
| High-ray reference budget | $200,000$ rays |
| Retained-path policy | Top-$k$ energy |
| Retained-path count | $5,000$ paths |
| Stored data type | float32 |
| Shoebox geometry range | Length: $4.0$-$14.0$ m;  Width: $3.0$-$10.0$ m;  Height: $2.4$-$4.5$ m |
| Corridor geometry range | Length: $8.0$-$24.0$ m;  Width: $1.8$-$4.0$ m;  Height: $2.4$-$4.0$ m |
| Source and receiver height range | $1.2$-$1.8$ m |
| Placement constraints | Minimum wall margin: $0.5$ m;  Minimum floor margin: $0.5$ m;  Minimum ceiling margin: $0.3$ m;  Source-receiver distance: $1.0$-$10.0$ m |
| Quality-control thresholds | Maximum onset mismatch: $2.0$ ms;  Minimum total energy: $1\*10^{-10}$ m;  Non-empty file path required;  Maximum retained-path file size: $128$ MB |

*Figure 6: Dataset split configuration summary*

| **Split** | **Count** | **Seed** | **Geometry family** | **Placement regime** | **Material regime** |
| --- | --- | --- | --- | --- | --- |
| train | 500 | 1001 | Shoebox | interior\_random+mid\_pair | Mixed primary material distribution |
| valid | 60 | 1002 | Shoebox | interior\_random+mid\_pair | Mixed primary material distribution |
| test\_id | 60 | 1003 | Shoebox | interior\_random+mid\_pair | Mixed primary material distribution |
| test\_material\_shift | 40 | 1004 | Shoebox | interior\_random+mid\_pair | Shifted towards ceiling\_absorptive and asymmetric\_walls |
| test\_placement\_shift | 30 | 1005 | Shoebox | near\_corner+mid\_pair | Mixed primary material distribution |
| test\_geometry\_shift | 30 | 1006 | Corridor | near\_wall+far\_pair | Mixed primary material distribution |

**Appendix C. Learning Representation and Preprocessing Details**

**C.1 HOA Tensor Layout**

Stored impulse-response arrays used channel-first layout, $(C, T)$. Before model training, arrays were transposed to time-major layout, $(T, C)$, so that the neural network processed each example as a multichannel temporal sequence. For the primary dataset, third-order ambisonics produced $C=16$ channels, and the $48$ kHz, $3.0$ s impulse-response duration produced $T=144,000$ samples. After inference, predictions were transposed back to channel-first layout before export.

**C.2 Path-Feature Matrix Construction**

For path-conditioned variants, each scene’s retained path file was converted into a fixed-size feature matrix with shape $(K, F)$, where $K$ is the configured number of retained paths and $F$ is the number of learned path features. The learned features included listener direction, path distance, arrival time, frequency-band intensity values, and summed path energy. Identifier-like fields, including path rank and original path index, were excluded from the learned feature set. If fewer than $K$ paths were available, the feature matrix was zero-padded.

**C.3 Feature Transformations and Normalization**

Positive-valued path features were log-compressed before normalization using $log(1+x)$ after clamping negative values to zero. Signed directional components were not log-compressed. HOA signals were normalized channel-wise using training-set means and standard deviations, with separate statistics for low-ray inputs and high-ray targets. Path features were normalized feature-wise using statistics computed from the training path matrices. Validation and test examples used the same training-set statistics to avoid information leakage.

**C.4 Exported Output Format**

Predicted outputs were denormalized using the target HOA statistics and saved as time-domain HOA impulse responses in the same channel-first layout as the original simulated responses. This ensured that low-ray inputs, high-ray references, and denoised predictions shared a common representation for objective evaluation, preview rendering, and qualitative inspection.

**Appendix D. Model Architecture and Ablation Details**

**D.1 Temporal CNN Architecture**

The temporal CNN operated on the time-major HOA representation described in Appendix C. The network first applied a one-sample Conv1D projection to map the input sequence into a learned feature representation. The projected sequence was then passed through a stack of residual Conv1D blocks using same-padding temporal convolutions. The final layer projected the learned representation back to the HOA output channels.

*Figure 7: Main CNN Architecture Details*

| **Component** | **Value** |
| --- | --- |
| Input shape | $(144,000, 16)$ |
| Input projection | Conv1D, kernel size $1$, $32$ filters |
| Residual blocks | $6$ |
| Conv1D layers per block | $2$ |
| Kernel size | $9$ |
| Dilation schedule | $(1, 1, 2, 2, 4, 4)$ |
| Width schedule | $(32, 32, 64, 64, 32, 32)$ |
| Normalization | Layer normalization |
| Activation | ReLU |
| Output projection | Conv1D, kernel size $1$, $16$ filters |

**D.2 Residual-Prediction Formulation**

In the full-prediction formulation, the model directly predicted the normalized high-ray response: $\hat{x}\_{high}=f\_{θ}(x\_{low})$. In the residual-prediction formulation, the model predicted a correction term added to the normalized low-ray input: $\hat{x}\_{high}=x\_{low}+f\_{θ}(x\_{low})$. In the implementation, the residual formulation appeared in the model graph as a final addition layer.

**D.3 Path-Conditioned Branch**

The path-conditioned model used the path-feature matrix described in Appendix C as an auxiliary input. The path branch flattened this matrix, passed it through dense layers, repeated the resulting embedding across the temporal dimension, and concatenated it with the projected HOA features before the main temporal convolutional stack.

*Figure 8: Differences between HOA-only and path-conditioned model details*

| **Component** | **HOA-only model** | **Path-conditioned model** |
| --- | --- | --- |
| HOA Input Shape | $(144,000, 16)$ | $(144,000, 16)$ |
| Path Input Shape | None | $(128, 14)$ |
| Path Dense Widths | None | $(64, 32)$ |
| Path Embedding Size | None | $32$ |
| Trainable Parameters | $218,832$ | $347,152$ |
| Non-Trainable Parameters | $0$ | $0$ |

**Appendix E. Training Configuration and Checkpoint Selection**

**E.1 Training Objective**

The training objective was implemented as a custom tail-aware impulse-response loss. Although HOA inputs and targets were stored in normalized form, the loss first denormalized predictions and targets using the high-ray target statistics. The loss was therefore evaluated in raw HOA amplitude space rather than normalized network space.

The objective combined four terms: a time-weighted Huber waveform term, a late-window Huber term, multi-resolution STFT terms, and energy-decay terms. The Huber terms used δ=1.0. The time-weighted waveform term divided the impulse response into early, middle, and late regions, with larger weights assigned to later samples to emphasize reverberant-tail preservation.

*Figure 9: Training parameters*

| **Component** | **Value** |
| --- | --- |
| Loss class | Tail-aware RIR loss |
| Huber $δ$ | $1.0$ |
| Loss domain | Denormalized HOA amplitude space |
| Early region | $t<50$ ms |
| Mid region | $50\leq t<200$ ms |
| Late region | $t\geq 200$ ms |
| Late-window start | $80$ ms |
| Early weight | $1.0$ |
| Mid weight | $2.0$ |
| Late weight | $4.0$ |
| Waveform Huber weight | $1.0$ |
| Late-window Huber weight | $0.25$ |
| Full-response MR-STFT weight | $0.02$ |
| Late-window MR-STFT weight | $0.05$ |
| Full-response EDC weight | $0.01$ |
| Late-window EDC weight | $0.02$ |
| EDC floor | $-60$ dB |

**E.2 Multi-Resolution STFT Configuration**

The loss included multi-resolution STFT terms over the full impulse response and over the late window. For each configured resolution, the STFT magnitude loss combined a spectral-convergence term with a log-magnitude difference term. The configured STFT resolutions were:

*Figure 10: Frame information*

| **Frame length** | **Frame step** | **FFT length** |
| --- | --- | --- |
| 512 | 128 | 512 |
| 1024 | 256 | 1024 |
| 2048 | 512 | 2048 |

**E.3 Optimization Hyperparameters**

All model variants were trained using the same optimization settings unless otherwise noted. This design ensured that ablation comparisons reflected differences in model formulation rather than differences in training procedure.

*Figure 11: Hyperparameter tuning details*

| **Parameter** | **Value** |
| --- | --- |
| Random seed | $42$ |
| Batch size | $1$ |
| Maximum epochs | $100$ |
| Initial learning rate | $0.001$ |
| Early-stopping patience | $12$ |
| Early-stopping minimum delta | $1\*10^{-5}$ |
| Learning-rate reduction factor | $0.5$ |
| Learning-rate reduction patience | $6$ |
| Minimum learning rate | $1\*10^{-6}$ |
| Mixed precision | False |

**E.4 Model Variants**

The same training and evaluation framework was used for the full-prediction, residual-prediction, HOA-only, and path-conditioned variants. The path-conditioned model used the same temporal CNN backbone as the HOA-only model, with an auxiliary path-feature input branch.

*Figure 12: Model variant details*

| **Variant** | **HOA input** | **Path features** | **Output formulation** |
| --- | --- | --- | --- |
| HOA-only full prediction | Yes | No | Full prediction |
| HOA-only residual prediction | Yes | No | Residual prediction |
| Path-conditioned full prediction | Yes | Yes | Full prediction |
| Path-conditioned residual prediction | Yes | Yes | Residual prediction |

For the path-conditioned configuration, the retained path input used $K=128$ paths and $F=14$ path features per path. The learned path features were listener direction, distance, arrival time, octave-band or frequency-band intensity values, and summed path energy. Identifier-like fields were excluded from the learned feature set.

**E.5 Dataset Splits, Checkpoint Selection, and Evaluation Use**

The training pipeline used the dataset splits for distinct purposes. The training split was used for parameter updates, the validation split was used for monitoring training behavior and selecting checkpoints, and the test splits were reserved for held-out evaluation. This separation prevented test-set performance from influencing optimization or model selection.

*Figure 13: Dataset splits*

| **Split** | **Use in training pipeline** | **Purpose** |
| --- | --- | --- |
| train | Parameter updates | Fits model weights using paired low-ray and high-ray examples |
| valid | Validation and checkpoint selection | Monitors generalization during training and selects the retained checkpoint |
| test\_id | Held-out evaluation only | Measures in-distribution performance on unseen examples from the training distribution |
| test\_material\_shift | Held-out evaluation only | Measures robustness under altered material distributions |
| test\_placement\_shift | Held-out evaluation only | Measures robustness under shifted source-receiver placement conditions |
| test\_geometry\_shift | Held-out evaluation only | Measures robustness under a controlled geometry change |

For each run, the checkpoint retained for later evaluation was selected according to validation-set behavior. No test split was used for optimization, hyperparameter adjustment, or checkpoint selection.

**E.6 Inference and Export Procedure**

After training, the selected checkpoint was loaded for inference on the configured validation and test splits. The same training-set normalization statistics used during training were applied to the low-ray inputs and, when applicable, to the path-feature inputs. Model outputs were then passed through the export procedure described in Appendix C.4 before downstream evaluation.

**Appendix F. Extended Experiment Discussion**

*Figure 14: Relative L2 ratio and ΔSNR by model and test split*

| **Model variant** | **Split** | **Low rel. L2** | **Pred rel. L2** | **Rel. L2 ratio** | **ΔSNR** |
| --- | --- | --- | --- | --- | --- |
| hoa\_direct | test\_id | $0.504$ | $7.417$ | $0.0684$ | $−23.51$ |
| hoa\_direct | test\_material\_shift | $0.4918$ | $7.149$ | $0.06926$ | $−23.32$ |
| hoa\_direct | test\_placement\_shift | $0.502$ | $7.578$ | $0.06611$ | $−23.81$ |
| hoa\_direct | test\_geometry\_shift | $0.6081$ | $7.96$ | $0.07709$ | $−22.45$ |
| hoa\_residual | test\_id | $0.504$ | $0.518$ | $0.9717$ | $-0.2494$ |
| hoa\_residual | test\_material\_shift | $0.4918$ | $0.5061$ | $0.9727$ | $-0.2404$ |
| hoa\_residual | test\_placement\_shift | $0.502$ | $0.5175$ | $0.9687$ | $-0.2767$ |
| hoa\_residual | test\_geometry\_shift | $0.6081$ | $0.6257$ | $0.9725$ | $-0.2426$ |
| path\_direct | test\_id | $0.504$ | $1.304$ | $0.3867$ | $-8.595$ |
| path\_direct | test\_material\_shift | $0.4918$ | $1.336$ | $0.368$ | $-8.847$ |
| path\_direct | test\_placement\_shift | $0.502$ | $1.262$ | $0.3963$ | $-8.361$ |
| path\_direct | test\_geometry\_shift | $0.6081$ | $1.36$ | $0.4481$ | $-7.263$ |
| path\_residual | test\_id | $0.504$ | $0.5134$ | $0.9808$ | $-0.1684$ |
| path\_residual | test\_material\_shift | $0.4918$ | $0.5018$ | $0.9814$ | $-0.1639$ |
| path\_residual | test\_placement\_shift | $0.502$ | $0.5126$ | $0.9785$ | $-0.1888$ |
| path\_residual | test\_geometry\_shift | $0.6081$ | $0.6218$ | $0.9787$ | $-0.1871$ |

*Figure 15: Selected acoustic metrics by model and test split*

| **Model variant** | **Split** | **T20 error reduction** | **T30 error reduction** | **C50 error reduction** |
| --- | --- | --- | --- | --- |
| hoa\_direct | test\_id | $0.03565$ | $0.01955$ | $0.3577$ |
| hoa\_direct | test\_material\_shift | $0.0468$ | $0.03236$ | $0.5855$ |
| hoa\_direct | test\_placement\_shift | $0.02363$ | $0.0138$ | $0.3981$ |
| hoa\_direct | test\_geometry\_shift | $0.002297$ | $-0.01129$ | $-0.3618$ |
| hoa\_residual | test\_id | $-5.7018×10^{-8}$ | $-1.5699×10^{-6}$ | $-1.4370×10^{-6}$ |
| hoa\_residual | test\_material\_shift | $1.4025×10^{-6}$ | $2.1377×10^{-7}$ | $-1.1833×10^{-6}$ |
| hoa\_residual | test\_placement\_shift | $-4.2614×10^{-8}$ | $-2.6981×10^{-6}$ | $-1.2923×10^{-6}$ |
| hoa\_residual | test\_geometry\_shift | $3.7505×10^{-7}$ | $-1.2500×10^{-7}$ | $9.7198×10^{-7}$ |
| path\_direct | test\_id | $0.01227$ | $0.003433$ | $0.352$ |
| path\_direct | test\_material\_shift | $0.04079$ | $0.01943$ | $0.6115$ |
| path\_direct | test\_placement\_shift | $-0.04068$ | $-0.01729$ | $0.3434$ |
| path\_direct | test\_geometry\_shift | $-0.008802$ | $-0.01906$ | $-1.769$ |
| path\_residual | test\_id | $0.007588$ | $0.009869$ | $-0.0243$ |
| path\_residual | test\_material\_shift | $0.007036$ | $0.01102$ | $-0.01805$ |
| path\_residual | test\_placement\_shift | $0.006802$ | $0.009337$ | $-0.01081$ |
| path\_residual | test\_geometry\_shift | $0.002892$ | $0.00445$ | $-0.01908$ |

Positive values indicate reduced absolute error relative to the low-ray baseline, while negative values indicate increased error. The selected scalar metrics show that metric preservation was not uniform: decay-time metrics improved for several variants, especially T20 and T30, while clarity and definition metrics were more mixed.

**Appendix G. Asset Attributions**

*Figure 16: Asset Attribution Table*

| **Asset** | **Author** | **License** | **Link to asset** |
| --- | --- | --- | --- |
| Claps wav file | synthnisse | CC0 | https://freesound.org/s/509526/ |
| Kick wav file | Stereo Surgeon | CC0 | https://freesound.org/s/261331/ |
| Lofi wav file | holizna | CC0 | https://freesound.org/s/629167/ |
| Birds wav file | funzerker | CC0 | https://freesound.org/s/520672/ |
| LJ speech dataset | Morris, William | CC0 | https://keithito.com/LJ-Speech-Dataset |

**10. WORKS CITED**

[1] T. Potter, Z. Cvetković, and E. De Sena, “On the relative importance of visual and spatial audio rendering on VR immersion,” Frontiers in Signal Processing, vol. 2, 2022, doi: 10.3389/frsip.2022.904866.

[2] L. Savioja and U. P. Svensson, “Overview of geometrical room acoustic modeling techniques,” J. Acoust. Soc. Am., vol. 138, no. 2, pp. 708–730, 2015, doi: 10.1121/1.4926438.

[3] N. R. Tsingos, T. Funkhouser, A. Ngan, and I. Carlbom, “Modeling acoustics in virtual environments using the uniform theory of diffraction,” in Proc. 28th Annu. Conf. Comput. Graph. Interact. Tech. (SIGGRAPH), 2001, doi: 10.1145/383259.383323.

[4] L. Savioja, “Simulation-based auralization of room acoustics,” Acoustics Today, vol. 16, no. 4, p. 48, 2020, doi: 10.1121/at.2020.16.4.48.

[5] C. C. J. M. Hak, R. H. C. Wenmaekers, and L. C. J. van Luxemburg, “Measuring room impulse responses: Impact of the decay range on derived room acoustic parameters,” Acta Acust. united Acust., vol. 98, no. 6, pp. 907–915, 2012, doi: 10.3813/AAA.918574.

[6] J. T. Kajiya, “The rendering equation,” ACM SIGGRAPH Comput. Graph., vol. 20, no. 4, pp. 143–150, 1986, doi: 10.1145/15886.15902.

[7] S. Bako et al., “Kernel-predicting convolutional networks for denoising Monte Carlo renderings,” ACM Trans. Graph., vol. 36, no. 4, 2017, doi: 10.1145/3072959.3073708.

[8] F. Zotter and M. Frank, Ambisonics. Cham, Switzerland: Springer, 2019, doi: 10.1007/978-3-030-17207-7.

[9] J. Daniel, “Spatial sound encoding including near field effect: Introducing distance coding filters and a viable, new ambisonic format,” in Proc. AES 23rd Int. Conf., 2003.

[10] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, pp. 436–444, 2015, doi: 10.1038/nature14539.

[11] H. B. Fırat, L. Maffei, and M. Masullo, “3D sound spatialization with game engines: The virtual acoustics performance of a game engine and a middleware for interactive audio design,” Virtual Reality, vol. 26, 2022, doi: 10.1007/s10055-021-00589-0.

[12] C. L. Christensen, G. Koutsouris, and J. H. Rindel, “The ISO 3382 parameters: Can we simulate them? Can we measure them?,” in Proc. Int. Symp. Room Acoust. (ISRA), Toronto, ON, Canada, Jun. 9–11, 2013.

[13] M. Pharr, W. Jakob, and G. Humphreys, Physically Based Rendering: From Theory to Implementation, 3rd ed. Cambridge, MA, USA: Morgan Kaufmann, 2017.

[14] C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “AmbiX—A suggested ambisonics format,” in Proc. Ambisonics Symp. 2011, Lexington, KY, USA, 2011.

[15] D. Di Carlo, P. Tandeitnik, C. Foy, N. Bertin, A. Deleforge, and S. Gannot, “dEchorate: A calibrated room impulse response dataset for echo-aware signal processing,” EURASIP J. Audio Speech Music Process., vol. 2021, no. 1, 2021, doi: 10.1186/s13636-021-00229-0.

[16] P. Massé, T. Carpentier, O. Warusfel, and M. Noisternig, “Denoising directional room impulse responses with spatially anisotropic late reverberation tails,” Appl. Sci., vol. 10, no. 3, p. 1033, 2020, doi: 10.3390/app10031033.

[17] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning spatiotemporal features with 3D convolutional networks,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2015, doi: 10.1109/ICCV.2015.510.

[18] J. S. Bradley, “Review of objective room acoustics measures and future needs,” Applied Acoustics, vol. 72, no. 10, pp. 713–720, Oct. 2011, doi: 10.1016/j.apacoust.2011.04.004.

[19] N. K. Kalantari, S. Bako, and P. Sen, “A machine learning approach for filtering Monte Carlo noise,” ACM Trans. Graph., vol. 34, no. 4, 2015, doi: 10.1145/2766977.

[20] T. Vogels et al., “Denoising with kernel prediction and asymmetric loss functions,” ACM Trans. Graph., vol. 37, no. 4, 2018, doi: 10.1145/3197517.3201388.

[21] Q. Hou, Z. Li, C. S. Marshall, S. Panneer, and F. Liu, “Fast Monte Carlo rendering via multi-resolution sampling,” in Graphic Interface, 2021, doi: 10.48550/arXiv.2106.12802.

[22] J. B. Allen and D. A. Berkley, “Image method for efficiently simulating small-room acoustics,” J. Acoust. Soc. Am., vol. 65, no. 4, pp. 943–950, 1979, doi: 10.1121/1.382599.

[23] T. Funkhouser et al., “A beam tracing method for interactive architectural acoustics,” J. Acoust. Soc. Am., vol. 115, no. 2, pp. 739–756, 2004, doi: 10.1121/1.1641020.

[24] C. Schissler and D. Manocha, “Interactive sound propagation and rendering for large multi-source scenes,” ACM Trans. Graph., vol. 36, no. 4, 2017, doi: 10.1145/3072959.2943779.

[25] J. Merimaa and V. Pulkki, “Spatial impulse response rendering I: Analysis and synthesis,” J. Audio Eng. Soc., vol. 53, no. 12, pp. 1115–1127, 2005.

[26] V. Pulkki, “Directional audio coding in spatial sound reproduction and stereo upmixing,” in Proc. AES 28th Int. Conf., Piteå, Sweden, 2006.

[27] Y. Zang and Q. Kong, “GSound-SIR: A spatial impulse response ray-tracing and high-order ambisonic auralization Python toolkit,” arXiv preprint arXiv:2503.17866, 2025, doi: 10.48550/arXiv.2503.17866.

[28] H. Autio, N. G. Vardaxis, and D. Bard-Hagberg, “An iterative ray tracing algorithm to increase simulation speed while maintaining overall precision,” Acoustics, vol. 5, no. 1, pp. 320–342, 2023, doi: 10.3390/acoustics5010019.

[29] M. Pezzoli, D. Perini, A. Bernardini, F. Borra, F. Antonacci, and A. Sarti, “Deep prior approach for room impulse response reconstruction,” Sensors, vol. 22, no. 7, p. 2710, 2022, doi: 10.3390/s22072710.

[30] A. Ratnarajah, Z. Tang, R. Aralikatti, and D. Manocha, “MESH2IR: Neural acoustic impulse response generator for complex 3D scenes,” in Proc. 30th ACM Int. Conf. Multimedia, 2022, doi: 10.1145/3503161.3548253.

[31] J. Lin, G. Götz, and S. J. Schlecht, “Deep room impulse response completion,” arXiv preprint arXiv:2402.00859, 2024, doi: 10.48550/arXiv.2402.00859.

[32] J. Xia and W. Zhang, “Upmix B-format ambisonic room impulse responses using a generative model,” Appl. Sci., vol. 13, no. 21, p. 11810, 2023, doi: 10.3390/app132111810.

[33] P. J. Huber, “Robust estimation of a location parameter,” The Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73–101, 1964, doi: 10.1214/aoms/1177703732.

[34] Robert Kieser, Pall Reynisson, and Timothy J. Mulligan, “Definition of signal-to-noise ratio and its critical role in split-beam measurements,” ICES Journal of Marine Science, vol. 62, no. 1, pp. 123–130, 2005, doi: 10.1016/j.icesjms.2004.09.006.

[35] Blender Foundation, “Blender,” GitHub. Accessed: May 16, 2026. [Online]. Available: https://github.com/blender