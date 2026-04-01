
% \onecolumn

% \subsection{Additional Case Study: Long-Form Transcription Failure Analysis}
% \label{app:longform-failure}

% \noindent
% This appendix provides a qualitative comparison of two KV-cache compression methods on a long LibriSpeech audio sample
% (Chapter 13754, duration: 232.62\,s), highlighting a repetition failure mode observed under \textsc{SnapKV} and successful
% completion under \textsc{AudioKV}.

% \subsubsection{Observations and Discussion}
% \label{app:longform-failure:observations}

% \begin{itemize}
%   \item \textbf{\textsc{SnapKV} exhibits a severe repetition failure.} After reaching a long-range dependency clause (``future states shall recognize''), the model collapses into a high-frequency repetition loop, which dominates the remainder of the decoding output and leads to a very high WER.
%   \item \textbf{\textsc{AudioKV} preserves long-range coherence.} The model completes the passage with consistent semantic continuity and low error rate, including the full multi-clause sentence regarding resurrection and marriage.
%   \item \textbf{Implication for KV-cache compression.} The contrast suggests that compression strategies that better preserve global context (e.g., frequency-domain preservation) can reduce decoding instability in long-form ASR generation.
% \end{itemize}

% \onecolumn 
% \begin{tcolorbox}[appendixbox,title={Reference transcript (LibriSpeech Chapter 13754, 232.62\,s)}]
% \footnotesize
% \setlength{\parskip}{0.45em}
% \raggedright
% They knew no north, no south, no east, no west. They stood positively by the Constitution and would have nothing to do in the bloody strife between brothers, unless indeed they were summoned by the authority to which they had already once loyally responded to furnish men and arms for their country's need. Every reader of recent and current history has learned of her rapid growth, of her repeated appeals for the recognition to which she had so long been entitled in the sisterhood of states, of the prompt refusals with which her pleas were persistently met, though other territories with smaller and more illiterate populations, more restricted resources, and in every way weaker claims, were allowed to assume the habiliments of maturity, while Utah, lusty, large and strong, was kept in swaddling clothes. But the cries of the vigorous infant were at length heeded, and in answer to the seventh appeal of the kind, Utah's star was added to the nation's galaxy. But let us turn more particularly to the history of the church itself. When Brigham Young passed from earth, he was mourned of the people as deeply as was Moses of Israel. And had he not proved himself a Moses, aye, and a Joshua, too. He had led the people into the land of holy promise and had divided unto them their inheritances. He was a man with clear title as one of the small brotherhood we call great. As carpenter, farmer, pioneer, capitalist, financier, preacher, apostle, prophet, in everything he was a leader among men. Even those who opposed him in politics and in religion respected him for his talents, his magnanimity, his liberality and his manliness, and years after his demise, men who had refused him honor while alive, brought their mites and their gold to erect a monument of stone and bronze to the memory of this man who needs it not. With his death closed another epoch in the history of his people, and a successor arose, one who was capable of leading and judging under the changed conditions. That plural marriage has been practised by a limited proportion of the people under sanction of church ordinance has never since the introduction of the system been denied. But that plural marriage is a vital tenet of the church is not true. What the Latter Day Saints call celestial marriage is characteristic of the church and is in very general practise, but of celestial marriage, plurality of wives was an incident, never an essential. Yet the two have often been confused in the popular mind. We believe in a literal resurrection and an actual hereafter, in which future state shall be recognized every sanctified and authorized relationship existing here on earth, of parent and child, brother and sister, husband and wife. We believe further that contracts as of marriage, to be valid beyond the veil of mortality, must be sanctioned by a power greater than that of earth. With the seal of the holy priesthood upon their wedded state, these people believe implicitly in the perpetuity of that relationship on the far side of the grave. They marry not with the saddening limitation until death do you part, but for time and for all eternity. To them marriage is not, can never be, a civil compact alone. Its significance reaches beyond the grave. Its obligations are eternal, and the Latter Day Saints are notable for the sanctity with which they invest the marital state. It has been my privilege to tread the soil of many lands, to observe the customs and study the habits of more nations than one, and I have yet to find the place and meet the people where and with whom the purity of man and woman is held more precious than among the maligned Mormons in the mountain valleys of the West.
% \end{tcolorbox}

% \subsubsection{Qualitative Comparison}
% \label{app:longform-failure:qualitative-comparison}

% \begin{tcolorbox}[appendixboxalt,title={\textsc{SnapKV} decoding output (repetition failure)}]
% \footnotesize
% \setlength{\parskip}{0.45em}
% \raggedright

% \textit{Context before failure (excerpt):}

% They knew no north, no south, no east, no west. They stood positively by the Constitution and would have nothing to do in the bloody strife between brothers, unless indeed they were summoned by the authority to which they had already once loyally responded to furnish men and arms for the country's need. Every reader of recent and current history has learned of her rapid growth, of her repeated appeals for the recognition to which she had so long been entitled in the sisterhood of states, of the prompt refusals with which her pleas were persistently met, though other territories with smaller and more illiterate populations, more restricted resources, and in every way weaker claims, were allowed to assume the habiliments of maturity, while Utah, lusty, large and strong, was kept in swaddling clothes. But the cries of the vigorous infant were at length heeded, and in answer to the seventh appeal of the kind, Utah's star was added to the nation's galaxy. But let us turn more particularly to the history of the church itself. When Brigham Young passed from earth, he was mourned of the people as deeply as was Moses of Israel, and had he not proved himself a Moses, aye, and a Joshua, too? He had led the people into the land of holy promise, and had divided it among them. He was a man with clear title as one of the small brotherhood we call great, as carpenter, farmer, pioneer, capitalist, financier, preacher, apostle, prophet, and in everything he was a leader among men.

% \medskip
% \textbf{Failure onset:} \emph{We believe in a literal resurrection, and in an actual hereafter, in which future states shall recognize}

% \medskip
% \textbf{Repetition loop (truncated):}
% \begin{quote}\ttfamily\footnotesize
% every future state shall recognize\\
% every future state shall recognize\\
% every future state shall recognize\\
% \emph{[repeats hundreds of times]}
% \end{quote}
% \end{tcolorbox}

% \subsubsection{\textsc{AudioKV} Decoding Output (Successful Completion)}
% \label{app:longform-failure:audiokv-output}

% \begin{tcolorbox}[appendixbox,title={\textsc{AudioKV} decoding output (successful completion)}]
% \footnotesize
% \setlength{\parskip}{0.45em}
% \raggedright

% They knew no north, no south, no east, no west. They stood positively by the Constitution and would have nothing to do in the bloody strife between brothers, unless indeed they were summoned by the authority to which they had already once loyally responded to furnish men and arms for the country's need. Every reader of recent and current history has learned of her rapid growth, of her repeated appeals for the recognition to which she had so long been entitled in the sisterhood of states, of the prompt refusals with which her pleas were persistently met, though other territories with smaller and more illiterate populations, more restricted resources, and in every way weaker claims, were allowed to assume the habiliments of maturity, while Utah, lusty, large, and strong, was kept in swaddling clothes. But the cries of the vigorous infant were at length heeded, and in answer to the seventh appeal of the kind, Utah's star was added to the nation's galaxy. But let us turn more particularly to the history of the church itself. When Brigham Young passed from earth, he was mourned of the people as deeply as was Moses of Israel, and had he not proved himself a Moses, aye, and a Joshua, too? He had led the people into the land of holy promise, and had divided unto them their inheritances. He was a man with clear title as one of the small brotherhood we call great, as carpenter, farmer, pioneer, capitalist, financier, preacher, apostle, prophet, and everything he was a leader among men, even those who opposed him in politics and in religion, respected him for his talents, his magnanimity, his liberality, and his manliness, and years after his demise, men who had refused him honor while alive, brought their mites and their gold to erect a monument of stone and bronze to the memory of this man who needs it not.

% \medskip
% We believe in a literal resurrection and an actual hereafter, in which future states shall be recognized every sanctified and authorized relationship existing here on earth, of parent and child, brother and sister, husband and wife. We believe further that contracts as of marriage, to be valid beyond the veil of mortality, must be sanctioned by a power greater than that of earth. With the seal of the holy priesthood upon their wedded state, these people believe implicitly in the perpetuity of that relationship on the far side of the grave. They marry not with the saddening limitation until death do you part, but for time and for all eternity. To them marriage is not, can never be, a civil compact alone. Its significance reaches beyond the grave. Its obligations are eternal, and the Latter Day Saints are notable for the sanctity with which they invest the marital state. It has been my privilege to tread the soil of many lands, to observe the customs and study the habits of more nations than one, and I have yet to find the place and meet the people where and with whom the purity of man and woman is held more precious than among the maligned Mormons in the mountain valleys of the West.
% \end{tcolorbox}


% \subsubsection{Discussion}
% \label{app:longform-failure:discussion}

% This case study highlights that long-form ASR robustness is determined not only by average token-level accuracy, but also by
% decoding stability under long-range dependencies. In the \textsc{SnapKV} output, a localized phrase transition triggers a
% self-reinforcing repetition loop, indicating that compressed KV representations may lose sufficient global context to recover
% from near-degenerate next-token distributions. Once entered, this loop dominates the remainder of generation and causes a
% disproportionate WER increase despite reasonable prefix quality.

% In contrast, \textsc{AudioKV} preserves coherent continuation through the same semantic region and completes the paragraph
% without collapse, suggesting that frequency-domain-aware retention better maintains discourse-level constraints over long
% horizons. Qualitatively, the difference is not a small wording variation but a mode-level behavioral divergence: one method
% fails catastrophically while the other remains stable.

% These observations suggest two practical implications for KV-cache compression in speech generation: (1) evaluation should
% include long-context stability metrics (e.g., repetition-loop incidence, late-stage divergence) in addition to aggregate WER,
% and (2) design choices that preserve globally informative components can meaningfully improve robustness in long-form decoding.

% \subsubsection{Failure Case Visualization Figure}
% \label{app:longform-failure:figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{plot/case_study/failure_case_visualization.pdf}
  \caption{Qualitative visualization of long-form decoding behavior. \textsc{SnapKV} falls into a repetition loop after the trigger phrase, while \textsc{AudioKV} remains coherent and completes the passage.}
  \label{fig:failure-case-visualization}
\end{figure}
