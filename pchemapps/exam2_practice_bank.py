"""Question bank for the CHE 341 Exam 2 scaffolded practice page.

Each ladder starts at a general chemistry anchor and climbs to exam level.
Question ids are stable keys in the response log: edit text freely, but give
a rewritten question a new id so old responses stay interpretable.

Fields: id, prompt, choices (list), answer (index into choices), explain.
Prompts and explanations are Streamlit markdown ($...$ for math).
"""

LADDERS = [
    {
        "title": "1. Gases, units, and intrinsic vs. extrinsic",
        "questions": [
            {
                "id": "gas-01",
                "prompt": r"Which pair of quantities are **both intrinsic** (independent of how much material you have)?",
                "choices": [
                    r"$V$ and $U$",
                    r"$T$ and $\overline{C}_V$",
                    r"$C_V$ and $T$",
                    r"$H$ and $\overline{V}$",
                ],
                "answer": 1,
                "explain": r"Intrinsic quantities do not scale with amount: $T$, $P$, and molar quantities like $\overline{C}_V = C_V/n$ and $\overline{V} = V/n$. Extrinsic quantities ($V$, $U$, $H$, $C_V$) double when you double the sample.",
            },
            {
                "id": "gas-02",
                "prompt": r"A 5.00 g sample of a gas with molar mass 44.0 g/mol has $C_V = 3.24$ J/K. What is $\overline{C}_V$?",
                "choices": [
                    r"$0.368$ J mol$^{-1}$ K$^{-1}$",
                    r"$0.648$ J mol$^{-1}$ K$^{-1}$",
                    r"$28.5$ J mol$^{-1}$ K$^{-1}$",
                    r"$143$ J mol$^{-1}$ K$^{-1}$",
                ],
                "answer": 2,
                "explain": r"$n = 5.00\ \text{g} / 44.0\ \text{g mol}^{-1} = 0.1136$ mol. $\overline{C}_V = C_V/n = 3.24\ \text{J K}^{-1} / 0.1136\ \text{mol} = 28.5$ J mol$^{-1}$ K$^{-1}$. Dividing an extrinsic quantity by $n$ makes it intrinsic.",
            },
            {
                "id": "gas-03",
                "prompt": r"A 2.00 L flask at 25.0 °C has $P = 0.750$ bar. How many moles of gas are present?",
                "choices": [
                    r"$0.0605$ mol",
                    r"$6.05 \times 10^{-4}$ mol",
                    r"$60.5$ mol",
                    r"$0.722$ mol",
                ],
                "answer": 0,
                "explain": r"$n = PV/RT = (0.750\ \text{bar})(2.00\ \text{L}) / [(0.08314\ \text{L bar mol}^{-1}\text{K}^{-1})(298.15\ \text{K})] = 0.0605$ mol. Pick the value of $R$ whose units match $P$ and $V$, and use $T$ in K.",
            },
            {
                "id": "gas-04",
                "prompt": r"You find the molar mass of an unknown gas from its density, $M = \rho RT/P$, measured in a glass bulb. Which measurement gives the most accurate molar mass?",
                "choices": [
                    r"The highest pressure you can reach, so the density is large and easy to weigh",
                    r"A low pressure, as low as you can go while still weighing the gas accurately",
                    r"Any pressure: the ideal gas law is exact for all gases",
                    r"A temperature just above where the gas condenses",
                ],
                "answer": 1,
                "explain": r"Gases behave ideally at low pressure (large $\overline{V}$), where molecules are far apart and their size and attractions are negligible. At high pressure the apparent molar mass drifts away from the true value.",
            },
            {
                "id": "gas-05",
                "prompt": r"At 300 bar and 298 K, H$_2$ has $Z = PV/(nRT) = 1.18$. What does $Z > 1$ tell you?",
                "choices": [
                    r"Attractions between molecules dominate, so the gas takes up less volume than an ideal gas",
                    r"The gas is ideal",
                    r"The finite size of the molecules (repulsions) dominates, so the gas takes up more volume than an ideal gas",
                    r"The gas is about to condense",
                ],
                "answer": 2,
                "explain": r"$Z > 1$ means $V$ is larger than $nRT/P$: repulsions (the $b$ term in van der Waals) dominate. $Z < 1$ means attractions (the $a$ term) dominate.",
            },
        ],
    },
    {
        "title": "2. The first law: heat, mechanical work, electrical work",
        "questions": [
            {
                "id": "fl-01",
                "prompt": r"A system absorbs 50 J of heat and the surroundings do 20 J of work on it. What is $\Delta U$?",
                "choices": [r"$-70$ J", r"$+30$ J", r"$-30$ J", r"$+70$ J"],
                "answer": 3,
                "explain": r"$\Delta U = q + w$. Heat into the system: $q = +50$ J. Work done on the system: $w = +20$ J. $\Delta U = +70$ J.",
            },
            {
                "id": "fl-02",
                "prompt": r"0.500 mol of He expands from 4.00 L to 10.00 L against a constant external pressure of 2.00 bar. What is $w$ in J?",
                "choices": [
                    r"$+1200$ J",
                    r"$-1200$ J",
                    r"$-12.0$ J",
                    r"$0$ J",
                ],
                "answer": 1,
                "explain": r"$w_\text{m} = -P_\text{ext}\Delta V = -(2.00\ \text{bar})(10.00 - 4.00)\ \text{L} = -12.0$ L bar $= -1200$ J. Negative: the gas does work on the surroundings.",
            },
            {
                "id": "fl-03",
                "prompt": r"A current of 0.500 A runs through a 10.0 Ω resistor inside a **rigid** steel can of gas for 30.0 s. What are $w_\text{m}$ and $w_\text{e}$?",
                "choices": [
                    r"$w_\text{m} = 0$, $w_\text{e} = +75.0$ J",
                    r"$w_\text{m} = +75.0$ J, $w_\text{e} = 0$",
                    r"$w_\text{m} = 0$, $w_\text{e} = -75.0$ J",
                    r"$w_\text{m} = -75.0$ J, $w_\text{e} = +75.0$ J",
                ],
                "answer": 0,
                "explain": r"Rigid walls: $dV = 0$, so $w_\text{m} = -\int P_\text{ext}\,dV = 0$. Electrical work done on the system: $w_\text{e} = I^2 R t = (0.500\ \text{A})^2(10.0\ \Omega)(30.0\ \text{s}) = +75.0$ J.",
            },
            {
                "id": "fl-04",
                "prompt": r"The same can is insulated (adiabatic walls). The 75.0 J of electrical work raises the gas from 25.00 °C to 31.25 °C. What is $C_V$ of the gas sample?",
                "choices": [
                    r"$2.40$ J/K",
                    r"$0.0833$ J/K",
                    r"$12.0$ J/K",
                    r"Cannot be determined because $q = 0$",
                ],
                "answer": 2,
                "explain": r"Adiabatic: $q = 0$. Rigid: $w_\text{m} = 0$. So $\Delta U = w_\text{e} = 75.0$ J. At constant $V$, $C_V \approx \Delta U/\Delta T = 75.0\ \text{J} / 6.25\ \text{K} = 12.0$ J/K.",
            },
            {
                "id": "fl-05",
                "prompt": r"An immersion heater warms 0.300 kg of water in a mug at constant $P$: $\Delta H = +75.3$ kJ, while the heater does $w_\text{e} = +82.0$ kJ of electrical work. What is $q$, and what does its sign mean?",
                "choices": [
                    r"$q = +75.3$ kJ: heat flows into the water",
                    r"$q = -6.7$ kJ: heat flows from the water to the room",
                    r"$q = +157.3$ kJ: heat flows into the water",
                    r"$q = 0$: the mug is adiabatic",
                ],
                "answer": 1,
                "explain": r"At constant $P$, $\Delta H = q + w_\text{non-mech}$, so $q = \Delta H - w_\text{e} = 75.3 - 82.0 = -6.7$ kJ. Not all of the electrical work stays in the water; some leaks to the room as heat.",
            },
            {
                "id": "fl-06",
                "prompt": r"An unknown gas has $\overline{C}_V = 89.7$ J mol$^{-1}$ K$^{-1}$ $= 10.8\,R$ near room temperature. What can you conclude?",
                "choices": [
                    r"It is monatomic ($\tfrac{3}{2}R$)",
                    r"It is diatomic with no vibrations active ($\tfrac{5}{2}R$)",
                    r"It is a polyatomic molecule with several vibrations contributing",
                    r"It is not behaving as a gas",
                ],
                "answer": 2,
                "explain": r"Translation gives $\tfrac{3}{2}R$ and rotation adds $R$ (linear) or $\tfrac{3}{2}R$ (nonlinear). Anything far above $3R$ means many vibrational levels are accessible at this temperature, which requires a large polyatomic molecule.",
            },
        ],
    },
    {
        "title": "3. State functions and paths",
        "questions": [
            {
                "id": "sf-01",
                "prompt": r"Hess's law lets you add $\Delta H$ values for any sequence of steps. Why?",
                "choices": [
                    r"Because $q$ is the same for every path",
                    r"Because $H$ is a state function (a real thermodynamic variable): $\Delta H$ depends only on the initial and final states",
                    r"Because all reactions occur at constant volume",
                    r"Because $\Delta H = \Delta U$ for every reaction",
                ],
                "answer": 1,
                "explain": r"$U$, $H$, $S$, $T$, $P$, $V$ are state functions, real thermodynamic variables: each has a definite value for a given equilibrium state. $q$ and $w$ are not thermodynamic variables. A system does not *have* heat or work; they describe a process and depend on how the change is carried out.",
            },
            {
                "id": "sf-02",
                "prompt": r"An ideal gas goes from state 1 to state 2 by two different paths, A and B. Which must be the same for both paths?",
                "choices": [
                    r"$q$ and $w$",
                    r"$q$ only",
                    r"$\Delta U$ and $\Delta S$ of the system",
                    r"$\Delta S$ of the surroundings",
                ],
                "answer": 2,
                "explain": r"$\Delta U$ and $\Delta S$ of the system depend only on the endpoints. $q$, $w$, and $\Delta S_\text{surr}$ depend on the path.",
            },
            {
                "id": "sf-03",
                "prompt": r"1.00 mol of N$_2$ (ideal gas) expands **irreversibly** against $P_\text{ext} = 1.00$ bar from 10.0 bar to 1.00 bar at a constant 300 K. How do you find $\Delta S$ of the gas?",
                "choices": [
                    r"$\Delta S = q/T$ using the actual heat of the irreversible process",
                    r"$\Delta S = 0$ because the temperature did not change",
                    r"Imagine a reversible isothermal path between the same two states: $\Delta S = \int dq_\text{rev}/T = nR\ln(V_\text{f}/V_\text{i})$",
                    r"$\Delta S = \Delta H/T$",
                ],
                "answer": 2,
                "explain": r"$dS = dq_\text{rev}/T$ only along a reversible path. Since $S$ is a state function, any reversible path with the same endpoints gives the system's $\Delta S$: here $nR\ln(10.0) = 19.1$ J/K.",
            },
            {
                "id": "sf-04",
                "prompt": r"Why is $\Delta U = 0$ for an **ideal** gas at constant temperature, whatever the change in volume?",
                "choices": [
                    r"Ideal gas particles do not interact, so their energy depends only on $T$, not on how far apart they are",
                    r"Because $q = 0$ for every isothermal process",
                    r"Because $w = 0$ for every isothermal process",
                    r"Because the translational energy levels do not depend on volume",
                ],
                "answer": 0,
                "explain": r"For an ideal gas $U$ is kinetic and internal (rotation, vibration) energy only, which depends on $T$. With attractions between molecules, pulling them apart would change $U$ even at constant $T$.",
            },
            {
                "id": "sf-05",
                "prompt": r"1.000 mol of methane is heated at 1 bar from 300.0 K to 600.0 K with $\overline{C}_P = 14.2 + 0.0750\,T$ (J mol$^{-1}$ K$^{-1}$). What is $\Delta S$?",
                "choices": [
                    r"$\overline{C}_P(600\ \text{K}) \ln(600/300) = 41.0$ J/K",
                    r"$\int (14.2/T + 0.0750)\,dT = 14.2\ln(600/300) + 0.0750(300) = 32.3$ J/K",
                    r"$\Delta H/T_\text{f} = 24.0$ J/K",
                    r"$0$, because pressure is constant",
                    r"It cannot be found: $\Delta S$ depends on whether the heating was done reversibly or irreversibly",
                ],
                "answer": 1,
                "explain": r"$dS = dq_\text{rev}/T = \overline{C}_P\,dT/T$ at constant $P$. With a temperature-dependent heat capacity, integrate each term: $\int a/T\,dT = a\ln(T_\text{f}/T_\text{i})$ and $\int b\,dT = b\,\Delta T$. $S$ is a state function, so $\Delta S$ of the methane is the same however it was heated: calculate it along a reversible path between the same endpoints. What depends on the path is $\Delta S_\text{surr}$ (and so $\Delta S_\text{universe}$).",
            },
        ],
    },
    {
        "title": "4. Reversible vs. irreversible",
        "questions": [
            {
                "id": "rev-01",
                "prompt": r"Which process is (approximately) **reversible**?",
                "choices": [
                    r"A hot metal block dropped into cold water",
                    r"A gas warmed very slowly so the surroundings are always at the same temperature as the gas",
                    r"A gas expanding into a vacuum",
                    r"A balloon popping",
                ],
                "answer": 1,
                "explain": r"Reversible means always at equilibrium with the surroundings: no finite temperature or pressure differences. A tiny change in conditions could run it backward.",
            },
            {
                "id": "rev-02",
                "prompt": r"1.00 mol of ideal gas in an insulated container expands into a vacuum, doubling its volume. Which is correct?",
                "choices": [
                    r"$q = 0$, $w = 0$, $\Delta S = 0$",
                    r"$q = 0$, $w = 0$, $\Delta S = R\ln 2 = +5.76$ J/K",
                    r"$q > 0$, $w < 0$, $\Delta S = q/T$",
                    r"$q = 0$, $w < 0$, $\Delta S < 0$",
                ],
                "answer": 1,
                "explain": r"No heat flows and nothing pushes back, so $q = w = 0$ and $\Delta U = 0$. But $\Delta S$ is not $q/T$ here: use the reversible isothermal path between the same states, $\Delta S = nR\ln(V_\text{f}/V_\text{i}) = R\ln 2$.",
            },
            {
                "id": "rev-03",
                "prompt": r"For the N$_2$ expansion (1.00 mol, 10.0 bar to 1.00 bar against $P_\text{ext} = 1.00$ bar, 300 K), $q = +2245$ J and $\Delta S_\text{sys} = +19.1$ J/K. What is $\Delta S_\text{universe}$?",
                "choices": [
                    r"$0$",
                    r"$+19.1$ J/K",
                    r"$+26.6$ J/K",
                    r"$+11.7$ J/K",
                ],
                "answer": 3,
                "explain": r"The surroundings stay at 300 K and lose 2245 J: $\Delta S_\text{surr} = q_\text{surr}/T = -2245/300 = -7.48$ J/K. $\Delta S_\text{universe} = 19.1 - 7.5 = +11.7$ J/K > 0, so the process is irreversible.",
            },
            {
                "id": "rev-04",
                "prompt": r"The same N$_2$ expansion is done **reversibly** at 300 K. Compared to the irreversible version, which is true?",
                "choices": [
                    r"$\Delta S_\text{sys}$ is the same; $q$ is larger; $\Delta S_\text{universe} = 0$",
                    r"$\Delta S_\text{sys}$ is larger; $q$ is the same",
                    r"$\Delta S_\text{sys} = 0$ because the process is reversible",
                    r"Everything is the same because the endpoints are the same",
                ],
                "answer": 0,
                "explain": r"$\Delta S_\text{sys}$ is a state function: still $+19.1$ J/K. Irreversibly, $w = -P_\text{ext}\Delta V$ (the orange rectangle). Reversibly, $P_\text{ext}$ matches the gas pressure the whole way, so the gas does much more work (the whole area under the curve). $\Delta U = 0$ for an ideal gas at constant $T$, so more work done means more heat absorbed from the surroundings: $q > 2245$ J. Reversible means $\Delta S_\text{universe} = 0$, so $\Delta S_\text{surr} = -19.1$ J/K.",
                "figure": "rev_vs_irrev_pv.png",
            },
            {
                "id": "rev-06",
                "prompt": r"1 mol O$_2$ is warmed from 273 K to 353 K at 1 bar in one of three ways: (A) in a bath whose temperature is raised slowly, (B) by dropping it into a bath held at 353 K, or (C) with an electric heater inside an insulated container. Which is true?",
                "choices": [
                    r"$\Delta S_\text{sys}$ is the same for all three; $\Delta S_\text{universe} = 0$ for A and $> 0$ for B and C",
                    r"$\Delta S_\text{sys}$ is larger for B and C",
                    r"$\Delta S_\text{universe} > 0$ for all three",
                    r"$\Delta S_\text{surr}$ is the same for all three",
                    r"$\Delta S_\text{universe} = 0$ for C, because no heat comes from the surroundings",
                ],
                "answer": 0,
                "explain": r"Same endpoints, so the same $\Delta S_\text{sys} = \overline{C}_P\ln(353/273) = +7.48$ J/K for all three. A: the bath is always at the gas temperature, so $\Delta S_\text{surr} = -7.48$ J/K and $\Delta S_\text{universe} = 0$. B: the 2328 J of heat leaves a bath at 353 K: $\Delta S_\text{surr} = -6.60$ J/K, so $\Delta S_\text{universe} = +0.88$ J/K. C: the heater does electrical work and no heat leaves the surroundings, so $\Delta S_\text{surr} = 0$ and $\Delta S_\text{universe} = +7.48$ J/K, the most irreversible of the three.",
            },
        ],
    },
    {
        "title": "5. Entropy of phase changes and reactions",
        "questions": [
            {
                "id": "rxn-01",
                "prompt": r"Without a data table, what is the sign of $\Delta_\text{r}S^\circ$ for 2 H$_2$(g) + O$_2$(g) → 2 H$_2$O(l)?",
                "choices": [
                    r"Positive: more bonds form",
                    r"Negative: 3 mol of gas become liquid",
                    r"Zero: atoms are conserved",
                    r"Cannot be predicted",
                ],
                "answer": 1,
                "explain": r"Gases have far more accessible (translational) states than liquids. Losing moles of gas almost always makes $\Delta_\text{r}S^\circ$ negative.",
            },
            {
                "id": "rxn-02",
                "prompt": r"At benzene's normal boiling point, 80.1 °C, $\Delta_\text{vap}H = 30.7$ kJ/mol. What is $\Delta_\text{vap}S$?",
                "choices": [
                    r"$383$ J mol$^{-1}$ K$^{-1}$: $\Delta S = \Delta H/T_\text{b}$, and benzene boils at 80.1 °C",
                    r"$86.9$ J mol$^{-1}$ K$^{-1}$: boiling at the normal boiling point follows a reversible path, so $\Delta S = q/T$ with $q = \Delta H$",
                    r"$103$ J mol$^{-1}$ K$^{-1}$: tabulated $\Delta H$ values are standard values, so use the standard temperature, 298.15 K",
                    r"$86.9$ J mol$^{-1}$ K$^{-1}$: $\Delta S = \Delta H/T$ for every process at constant $P$",
                ],
                "answer": 1,
                "explain": r"At 1 bar and 80.1 °C, liquid and vapor are at equilibrium, so boiling there follows a reversible path. At constant $P$ with no non-mechanical work, $q = \Delta H$. For 1 mol of benzene, $q = +30\,700$ J, so $\Delta S = q/T = 30\,700\ \text{J}/353.25\ \text{K} = +86.9$ J/K: $\Delta_\text{vap}S = +86.9$ J mol$^{-1}$ K$^{-1}$. Where the value comes from, roughly: 1 mol of liquid benzene is about 0.096 L, and 1 mol of vapor at 1 bar and 353 K is about 29 L. $R\ln(29/0.096) \approx 48$ J mol$^{-1}$ K$^{-1}$ is just the extra volume. The rest comes from molecules escaping the attractions that hold them in place in the liquid.",
            },
            {
                "id": "rxn-03",
                "prompt": r"For N$_2$(g) + 3 H$_2$(g) → 2 NH$_3$(g), $S^\circ$(N$_2$) = 191.6, $S^\circ$(H$_2$) = 130.7, and $S^\circ$(NH$_3$) = 192.8 J mol$^{-1}$ K$^{-1}$. What is $\Delta_\text{r}S^\circ$?",
                "choices": [
                    r"$+385.6$ J mol$^{-1}$ K$^{-1}$: elements in their standard states have $S^\circ = 0$, like $\Delta_\text{f}H^\circ$",
                    r"$-198.1$ J mol$^{-1}$ K$^{-1}$",
                    r"$+198.1$ J mol$^{-1}$ K$^{-1}$",
                    r"$-129.5$ J mol$^{-1}$ K$^{-1}$",
                ],
                "answer": 1,
                "explain": r"$\Delta_\text{r}S^\circ = \sum n_\text{p} S^\circ(\text{products}) - \sum n_\text{r} S^\circ(\text{reactants}) = 2(192.8) - [191.6 + 3(130.7)] = -198.1$ J mol$^{-1}$ K$^{-1}$. As expected, $\Delta S$ is negative when the moles of gas decrease: 4 mol of gas become 2. Unlike $\Delta_\text{f}H^\circ$, $S^\circ$ of an element is not zero: $S = 0$ only for a perfect crystal at 0 K.",
            },
            {
                "id": "rxn-04",
                "prompt": r"For the same reaction, $\Delta_\text{r}H^\circ = -91.8$ kJ/mol. A student says $\Delta_\text{r}S^\circ = \Delta_\text{r}H^\circ/T = -308$ J mol$^{-1}$ K$^{-1}$. What is wrong?",
                "choices": [
                    r"Nothing; the table values must be wrong",
                    r"They should have used $\Delta_\text{r}U^\circ$ instead of $\Delta_\text{r}H^\circ$",
                    r"$\Delta H/T$ gives $\Delta S$ of the system only at equilibrium, like a liquid boiling at its normal boiling point. This reaction, with 1 bar of each gas, is far from equilibrium (irreversible), so $\Delta_\text{r}H^\circ/T$ is not $\Delta_\text{r}S^\circ$",
                    r"They forgot to convert °C to K",
                ],
                "answer": 2,
                "explain": r"With no non-mechanical work (ordinary chemistry at constant $P$), $q = \Delta H$. For 1 mol of reaction as written (1 mol N$_2$ + 3 mol H$_2$ → 2 mol NH$_3$), $\Delta H = (1\ \text{mol})(-91.8\ \text{kJ/mol}) = -91.8$ kJ, so $q_\text{surr} = +91.8$ kJ and $\Delta S_\text{surr} = q_\text{surr}/T_\text{surr} = +91\,800\ \text{J}/298.15\ \text{K} = +308$ J/K. The student's 308 is the entropy change of the surroundings (sign flipped), not of the reaction. For the same 1 mol of reaction, $\Delta S_\text{sys} = -198.1$ J/K, from the $S^\circ$ table. The two are equal and opposite only at equilibrium ($\Delta S_\text{universe} = 0$).",
            },
            {
                "id": "rxn-05",
                "prompt": r"N$_2$ and H$_2$ react to form NH$_3$ at 298 K with 1 bar of each gas: $\Delta_\text{r}S^\circ = -198.1$ J mol$^{-1}$ K$^{-1}$ and $\Delta_\text{r}H^\circ = -91.8$ kJ/mol. What is $\Delta S_\text{universe}$ when 1.00 mol of H$_2$ is consumed?",
                "choices": [
                    r"$+36.6$ J/K: the reaction can proceed",
                    r"$+109.8$ J/K: the reaction can proceed",
                    r"$-66.0$ J/K: the reaction cannot proceed, because the moles of gas decrease",
                    r"$-168.7$ J/K: the reaction cannot proceed",
                ],
                "answer": 0,
                "explain": r"1.00 mol of H$_2$ is 1/3 mol of reaction (N$_2$ + 3 H$_2$ → 2 NH$_3$). For that much reaction: $q = \Delta H = (1/3\ \text{mol})(-91.8\ \text{kJ/mol}) = -30.6$ kJ, so $q_\text{surr} = +30.6$ kJ and $\Delta S_\text{surr} = +30\,600\ \text{J}/298.15\ \text{K} = +102.6$ J/K. $\Delta S_\text{sys} = (1/3\ \text{mol})(-198.1\ \text{J mol}^{-1}\text{K}^{-1}) = -66.0$ J/K. $\Delta S_\text{universe} = -66.0 + 102.6 = +36.6$ J/K > 0: the reaction can proceed even though the system's entropy drops, because the heat released raises the entropy of the surroundings more.",
            },
        ],
    },
    {
        "title": "6. Molecular entropy and concentration",
        "questions": [
            {
                "id": "mol-01",
                "prompt": r"Which has the higher molar entropy at 298 K: CH$_3$OH(g) or CH$_3$OH(l)?",
                "choices": [
                    r"CH$_3$OH(l): the liquid has hydrogen bonds",
                    r"CH$_3$OH(g): gas molecules have closely spaced translational and rotational levels, so far more states are accessible",
                    r"They are equal: same molecule",
                    r"CH$_3$OH(l): the liquid is denser, so more molecules fit in the same volume",
                ],
                "answer": 1,
                "explain": r"Molecules in the gas translate and rotate freely; in the liquid those motions are hindered by neighbors and hydrogen bonds. More accessible states means higher entropy: $S^\circ$ = 239.9 (g) vs. 126.8 (l) J mol$^{-1}$ K$^{-1}$.",
            },
            {
                "id": "mol-02",
                "prompt": r"Which has the higher molar entropy at 298 K: F$_2$(g) or Cl$_2$(g)?",
                "choices": [
                    r"F$_2$: lighter molecules move faster",
                    r"Cl$_2$: heavier and with a longer bond, so its translational and rotational levels are more closely spaced",
                    r"F$_2$: fluorine is more electronegative",
                    r"They are equal: both are homonuclear diatomic gases",
                ],
                "answer": 1,
                "explain": r"Translation: Cl$_2$ is much heavier (71 vs. 38 g/mol), and $q_\text{trans} \propto m^{3/2}$. Rotation: Cl$_2$ has a larger reduced mass and a longer bond (1.99 vs. 1.42 Å), so $\mu R^2$ is larger, $B$ and $\Theta_\text{rot}$ are smaller, and $q_\text{rot}$ is larger. Closer spacing means more accessible levels and higher entropy: $S^\circ$ = 223.1 (Cl$_2$) vs. 202.8 (F$_2$) J mol$^{-1}$ K$^{-1}$.",
            },
            {
                "id": "mol-03",
                "prompt": r"Which has the higher entropy: 1.00 mol of N$_2$ at 300 K and 1 bar, or 1.00 mol of N$_2$ at 300 K and 10 bar?",
                "choices": [
                    r"10 bar: the molecules collide more often",
                    r"1 bar: each molecule has 10 times more volume, so its translational levels are more closely spaced",
                    r"They are equal: same temperature",
                    r"10 bar: the molecules have more kinetic energy",
                ],
                "answer": 1,
                "explain": r"At 1 bar the gas fills 10 times the volume. Particle in a box: level spacing scales as $1/L^2$, so a larger volume means more accessible translational states. $\overline{S} = S^\circ - R\ln(P/1\ \text{bar})$, so the 10 bar gas is lower by $R\ln 10 = 19.1$ J mol$^{-1}$ K$^{-1}$.",
            },
            {
                "id": "mol-04",
                "prompt": r"O$_2$ in air has $P_{\text{O}_2} = 0.21$ bar. $S^\circ$(O$_2$) = 205.2 J mol$^{-1}$ K$^{-1}$ at 1 bar. What is $\overline{S}$ of O$_2$ in air at 298 K?",
                "choices": [
                    r"$192.2$ J mol$^{-1}$ K$^{-1}$",
                    r"$205.2$ J mol$^{-1}$ K$^{-1}$",
                    r"$43.1$ J mol$^{-1}$ K$^{-1}$",
                    r"$218.2$ J mol$^{-1}$ K$^{-1}$",
                ],
                "answer": 3,
                "explain": r"$\overline{S} = S^\circ - R\ln(P/1\ \text{bar}) = 205.2 - 8.314\ln(0.21) = 205.2 + 13.0 = 218.2$ J mol$^{-1}$ K$^{-1}$. Lower pressure means more volume per molecule, so higher entropy.",
            },
            {
                "id": "mol-06",
                "prompt": r"L-alanine(aq) ⇌ D-alanine(aq) has $\Delta_\text{r}H = 0$ and $\Delta_\text{r}S^\circ = 0$. Starting from pure L, why does the reaction stop at 50% L, 50% D?",
                "choices": [
                    r"D-alanine is less stable than L-alanine",
                    r"Heat released to the surroundings runs out",
                    r"The mixing (configurational) entropy is largest at 50:50; going further would unmix the solution and lower the entropy",
                    r"The reaction is too slow to go further",
                ],
                "answer": 2,
                "explain": r"With $\Delta_\text{r}H = 0$, $\Delta S_\text{surr} = 0$ and $\Delta S_\text{universe} = -R\ln([\text{D}]/[\text{L}])$. This is positive while [D] < [L] and zero at $Q_\text{rxn} = 1$. Concentration (configurational) entropy alone sets where the reaction stops.",
            },
        ],
    },
]


# Equation-sheet crops shown with each explanation (keys in pchemapps/exam2_sheet.CROPS)
SHEET_REFS = {
    "gas-01": ["C_molar"],
    "gas-02": ["C_molar"],
    "gas-03": ["R_lbar"],
    "gas-04": ["ideal_gas"],
    "gas-05": ["Z"],
    "fl-01": ["first_law"],
    "fl-02": ["w_mech", "L_bar"],
    "fl-03": ["w_mech_elec"],
    "fl-04": ["C_V"],
    "fl-05": ["dH_const_P"],
    "fl-06": ["U_ig"],
    "sf-01": ["hess"],
    "sf-03": ["dS_pressure"],
    "sf-04": ["U_ig"],
    "sf-05": ["dS_def", "int_dx_x"],
    "rev-02": ["dS_volume"],
    "rev-03": ["dS_surr_univ"],
    "rev-04": ["first_law_w_mech", "dS_surr"],
    "rev-06": ["dS_surr_univ"],
    "rxn-02": ["dS_def"],
    "rxn-03": ["hess"],
    "rxn-04": ["dH_const_P", "dS_surr"],
    "rxn-05": ["dS_surr_univ"],
    "mol-01": ["boltzmann_S"],
    "mol-03": ["particle_box", "S_pressure"],
    "mol-02": ["q_trans_rot", "rigid_rotor_B"],
    "mol-04": ["S_pressure"],
    "mol-06": ["S_rxn_Q"],
}

# Where to practice more, shown for questions missed on the first try.
# A4-A6: 26FA assignments. H&G: Hanson & Green book problems on A5/A6.
# M&S: McQuarrie & Simon extra practice listed on A6.
FOLLOW_UP = {
    "gas-01": "A4: intrinsic vs. extrinsic",
    "gas-02": "A4: Internal energy of ideal gases; A5 problem 1 (CO2)",
    "gas-03": "A4: Mystery gas",
    "gas-04": "A4: Mystery gas (which pressures?)",
    "gas-05": "Compressibility Factor app; Exam 1 gases",
    "fl-01": "A4: Types of work",
    "fl-02": "A4: Integral practice (work); Types of work (a)",
    "fl-03": "A4: Types of work; Python 1st Law activity (Dewar)",
    "fl-04": "A4: Python 1st Law activity (Dewar); A5 problem 1",
    "fl-05": "A4: Types of work; H&G 4.14",
    "fl-06": "A5 problem 1 (CO2) and ammonia heat capacity; A4: Mystery gas",
    "sf-01": "H&G 5.12",
    "sf-02": "A5: Gas Processes (why must ΔU be the same?)",
    "sf-03": "A6 problem 2 (Gas processes); M&S 6-8, 6-9, 6-14",
    "sf-04": "A4: Types of work (f, g)",
    "sf-05": "A5: ammonia ΔH; A6: SO2 heat capacity; M&S 6-27, 7-3, 7-4",
    "rev-01": "A6: O2 warmed slowly vs. dropped in a bath",
    "rev-02": "H&G 8.1; M&S 6-8",
    "rev-03": "A6: Gas processes II (O2 in a 353 K bath); M&S 6-13, 6-25, 6-26",
    "rev-04": "A4: Types of work (f); A6: comparing the two O2 processes",
    "rev-06": "A6: O2 problems; 2nd Law Calorimeter activity",
    "rxn-01": "H&G 9.2",
    "rxn-02": "H&G 9.2; M&S 7-40, 7-41",
    "rxn-03": "A6: entropy of SO2 from the data table",
    "rxn-04": "A6: 2nd Law Calorimeter activity; M&S 6-13, 6-25",
    "rxn-05": "A6: 2nd Law Calorimeter activity; M&S 6-26, 7-2",
    "mol-01": "H&G 7.7, 7.15",
    "mol-02": "H&G 7.7",
    "mol-03": "H&G 8.1; M&S 6-28",
    "mol-04": "A6: N2/O2 mixture; SO2 at 0.010 bar",
    "mol-06": "H&G 8.9; M&S 6-28, 6-30",
}


def all_questions():
    """Flat list of (ladder_index, ladder_title, question) in order."""
    out = []
    for i, ladder in enumerate(LADDERS):
        for q in ladder["questions"]:
            out.append((i, ladder["title"], q))
    return out
