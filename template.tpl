\documentclass[8pt,a4paper]{article}

\usepackage[a4paper,text={17.5cm,25.2cm},centering]{geometry}
\usepackage{lmodern}
\usepackage{amssymb,amsmath}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1.2ex}

\hypersetup
       {   pdfauthor = { Fredrik Bagge Carlson },
           pdftitle={ {{{:title}}} },
           colorlinks=TRUE,
           linkcolor=black,
           citecolor=blue,
           urlcolor=blue
       }

% Fonts =====================================0
\usepackage{unicode-math}
\usepackage{fontspec}
\setmainfont{Tex Gyre Pagella}[Scale=0.85]
\setmathfont{Tex Gyre Pagella Math}[Scale=MatchLowercase]
\setsansfont{TeX Gyre Heros}[Scale=MatchLowercase]
\setmonofont{DejaVu Sans}[Scale=MatchLowercase,StylisticSet={1,3}]
\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}



{{#:title}}
\title{ {{{ :title }}} }
{{/:title}}

{{# Fredrik Bagge Carlson}}
\author{ {{{  Fredrik Bagge Carlson }}} }
{{/ Fredrik Bagge Carlson}}

{{#:date}}
\date{ {{{ :date }}} }
{{/:date}}

{{ :highlight }}

\begin{document}

{{#:title}}\maketitle{{/:title}}

{{{ :body }}}

\end{document}
