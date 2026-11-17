#!/usr/bin/env wolframscript
(* Wolfram's lifetime experiments run with Wolfram's own code, in the Wolfram Language.

     wolframscript -file reproduce/wolfram_native.wls [phase ...]
     WOLFRAM_PHASES=phase,... wolfram -script reproduce/wolfram_native.wls        (kernel only, e.g. on a cluster)

   phases (default: verify replay detail timing):
     verify    his RuleArray code on every published array in tests/data/wolfram_rule_arrays.json
     replay    his seeded runs exactly as his figure code makes them:
                 lifetime 50, rules 4/146, 64 x 31: SeedRandom[234234 + i], i = 1..100 (figures 5 and 6 of the essay;
                   figure 5 shows runs {1, 2, 3, 4, 5, 6, 7, 90, 97, 89, 20, 13}) and SeedRandom[234241] (figure 3)
                 lifetime 30, And/Xor, 40 x 21: SeedRandom[234234] (figure 20, "1705 mutations")
     detail    the And/Xor run of figure 20 mutation by mutation (accepted, pattern-changing, lifetime-changing ...)
     sample    more seeds for the distribution of mutations to solution (N via WOLFRAM_SAMPLE, default 4096), with
               his RuleArray compiled (checked to give runs identical to the interpreted code)
     timing    seconds per candidate and per run on one kernel, his code as published (interpreted) and with his
               RuleArray function compiled by FunctionCompile (his Typed signature; the evolution itself unchanged)
   equal-time phases (benchmarks/equal_time.py measures STICA and numba the same way):
     loopcheck      adaptLoop (his AdaptRuleArray step in a loop with a deadline) makes exactly his run
     latency        one run at a time on one kernel (WOLFRAM_LATENCY_INTERP, _COMPILED runs)
     first          one run per subkernel started together: seconds to the first solution (WOLFRAM_FIRST_REPEATS)
     harvest        lifetime-50 runs on every subkernel for WOLFRAM_BUDGET seconds: every solution, hashed
     anytime        WOLFRAM_SCALE (default 10: 640 x 311, lifetime 500) for WOLFRAM_BUDGET seconds, runs restarted
                    after each solution; WOLFRAM_MODE interpreted (one kernel) or compiled (subkernels)
     reach          the same for each scale in WOLFRAM_SCALES, WOLFRAM_REACH_BUDGET seconds each
     ladder_timing  seconds per mutation at each scale, interpreted and compiled
   WOLFRAM_OUT sets the results file; WOLFRAM_CODE_FILE loads his code from a saved copy of the file.

   His code is not copied into this repository: it is loaded from the "copy code" file attached to figure 3 of the essay
   and only its definitions are evaluated (no plotting, no cloud calls). The file holds notebook boxes; each input cell is
   turned back into linear code by joining its tokens and parsed with ToExpression. Results go to
   reproduce/results/wolfram_native.json; reproduce/compare_wolfram.py compares them with STICA.
   Parallel phases use subkernels (environment variable WOLFRAM_KERNELS, default 8); run under a memory cap, e.g.
     systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 wolframscript -file reproduce/wolfram_native.wls *)

root = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
(* phases from the command line (wolframscript), or from WOLFRAM_PHASES, comma-separated (the kernel's -script mode
   passes no arguments) *)
phases = Which[Length[$ScriptCommandLine] > 1, Rest[$ScriptCommandLine],
   StringQ[Environment["WOLFRAM_PHASES"]], StringSplit[Environment["WOLFRAM_PHASES"], ","],
   True, {"verify", "replay", "detail", "timing"}];
outFile = Replace[Environment["WOLFRAM_OUT"],
   $Failed :> FileNameJoin[{root, "reproduce", "results", "wolfram_native.json"}]];
out = If[FileExistsQ[outFile], Import[outFile, "RawJSON"], <||>];     (* phases run separately accumulate *)
out["version"] = $Version;
out["date"] = DateString["ISODateTime"];
save[] := Export[outFile, out, "JSON"];

(* ---------------------------------------------------------------------------------------- Wolfram's definitions *)
codeURL = "https://content.wolfram.com/sites/43/2024/08/sw08202024arraysBimg3_copy.txt";
bannedHeads = {CloudImport, CloudObject, ResourceFunction, GraphicsGrid, Graphics, ListPlot, Labeled, Show, Print,
   SeedRandom, ParallelTable, Export, Import};
linear[b_] := StringReplace[StringRiffle[Cases[b, _String, {0, Infinity}], " "], "\[IndentingNewLine]" -> "\n"];
definitionQ[HoldComplete[e_]] := FreeQ[Unevaluated[e], Alternatives @@ bannedHeads] &&
   MatchQ[Unevaluated[e], HoldPattern[_SetDelayed | _Set | CompoundExpression[(_SetDelayed | _Set | Null) ..]]];
definitionQ[_] := False;
(* WOLFRAM_CODE_FILE: a saved copy of the same file, for hosts without internet (its sha256 is recorded below) *)
codeText = Import[Replace[Environment["WOLFRAM_CODE_FILE"], $Failed -> codeURL], "Text"];
boxes = Flatten[Replace[Cases[ToExpression[codeText, InputForm, HoldComplete], Cell[BoxData[b_], "Input", ___] :> b,
     Infinity], l_List :> l, {1}]];
held = Quiet[Check[ToExpression[linear[#], InputForm, HoldComplete], $Failed]] & /@ boxes;
defs = Select[held, definitionQ];
Scan[ReleaseHold, defs];
wanted = {"RuleArray", "MetaRuleArray", "RuleArrayStates", "RuleArrayLifetime", "LifetimeLoss", "LifetimeLossValue",
   "IndicesToRules", "MutateRuleArray", "AdaptRuleArray"};
loaded = Select[wanted, Length[DownValues[Evaluate[Symbol[#]]]] + Length[SubValues[Evaluate[Symbol[#]]]] > 0 &];
out["code"] = <|"url" -> codeURL, "sha256" -> Hash[codeText, "SHA256", "HexString"], "cells" -> Length[boxes],
   "definitions_loaded" -> Length[defs], "functions" -> loaded|>;
Print["loaded ", Length[defs], " definitions; functions: ", loaded];
If[Complement[wanted, loaded] =!= {}, Print["missing: ", Complement[wanted, loaded]]; Exit[1]];

opts1 = Sequence["Radius" -> 1, "Orientation" -> "Both"];
optsHalf = Sequence["Radius" -> 1/2, "Orientation" -> "Both"];
(* his calls, verbatim apart from the seed argument *)
lifetime50Run[seed_, lossFn_ : Automatic] := (SeedRandom[seed];
   With[{raHeight = 64, raWidth = 31, rules = {4, 146}, opts = opts1, loss = LifetimeLoss[50]},
    With[{lf = Replace[lossFn, Automatic :> loss[raHeight, raWidth, opts]]},
     AdaptRuleArray[ConstantArray[1, {raHeight, raWidth}], rules, 10000, lf, opts]]]);
andXorRun[seed_, lossFn_ : Automatic] := (SeedRandom[seed];
   With[{raHeight = 40, raWidth = 21, rules = {8, 6}, opts = optsHalf, loss = LifetimeLoss[30]},
    With[{lf = Replace[lossFn, Automatic :> loss[raHeight, raWidth, opts]]},
     AdaptRuleArray[ConstantArray[1, {raHeight, raWidth}], rules, 10000, lf, opts]]]);
(* his LifetimeLoss with his RuleArray function compiled (FunctionCompile of MetaRuleArray's specialised Function) *)
compiledLoss[target_, height_, width_, opts___] := With[
   {cf = FunctionCompile[MetaRuleArray[opts, "Function" -> "Lifetime"]]},
   Function[ra, LifetimeLossValue[cf[CenterArray[1, width], ra, height], target, height + 1]]];
(* one NestWhileList history {{array, loss}, ...}: mutations made, final loss, accepted mutations, loss changes *)
summary[res_] := <|"mutations" -> Length[res] - 1, "final_loss" -> ToString[Last[Last[res]]],
   "solved" -> Last[Last[res]] === 0,
   "accepted" -> Count[Partition[res[[All, 1]], 2, 1], {a_, b_} /; a =!= b],
   "loss_changes" -> Length[SplitBy[res, Last]] - 1,
   "loss_trajectory" -> (ToString /@ Last /@ (First /@ SplitBy[res, Last])),
   "distinct_arrays" -> Length[DeleteDuplicates[res[[All, 1]]]],
   "changed_cells" -> Count[Flatten[Last[res][[1]]], 2], "array" -> (StringJoin /@ Map[ToString, Last[res][[1]] - 1, {2}])|>;   (* rows of 0 (first rule) and 1 *)

(* ------------------------------------------------------------------------------------ for the equal-time phases *)
envNum[name_, default_] := ToExpression[Replace[Environment[name], $Failed -> ToString[default]]];
(* his lifetime-50 task scaled by s in both directions (s = 10: 640 x 311, lifetime 500), as benchmarks/equal_time.py *)
scaled[s_] := <|"height" -> 64 s, "width" -> If[s == 1, 31, 31 s + 1], "target" -> 50 s, "scale" -> s|>;
(* AdaptRuleArray's step, unchanged (MutateRuleArray, then his loss of IndicesToRules, kept if the loss does not
   rise), in a loop that keeps only the current array, where his NestWhileList keeps every array (1.6 MB each at
   640 x 311), and that stops at a wall-clock deadline. With restart a run that reaches zero loss is counted and starts
   again from ruleArray. Records each new best loss and each solution with its time and mutations. The loopcheck phase
   checks that it makes exactly the run AdaptRuleArray makes. *)
adaptLoop[ruleArray_, rules_, lossFn_, deadline_, t0_, restart_, maxIters_, opts___] :=
  Module[{cur = ruleArray, best, best0, m, lo, k = 0, steps = 0, bestEver, events, sols = {}},
   best0 = best = lossFn[IndicesToRules[ruleArray, rules, opts]];
   bestEver = best; events = {{AbsoluteTime[] - t0, ToString[best]}};
   While[AbsoluteTime[] < deadline && k < maxIters,
    m = MutateRuleArray[cur, Length[rules], opts];
    lo = lossFn[IndicesToRules[m, rules, opts]];
    k++; steps++;
    If[lo <= best, cur = m; best = lo];
    If[best < bestEver, bestEver = best; AppendTo[events, {AbsoluteTime[] - t0, ToString[best]}]];
    If[best == 0,
     AppendTo[sols, {AbsoluteTime[] - t0, steps}];
     If[restart, cur = ruleArray; best = best0; steps = 0, Break[]]]];
   <|"mutations" -> k, "events" -> events, "solutions" -> sols, "final" -> cur|>];
(* lifetime-50 runs (his AdaptRuleArray, compiled loss) one after another until the deadline; a run counts if it ends
   before the deadline. Solutions are kept as integers (one bit per cell) and hashed after the deadline. *)
harvestKernel[budget_, seedBase_] := Module[{sols = {}, j = 0, runs = 0, res, t0 = AbsoluteTime[], deadline},
   deadline = t0 + budget;                  (* each kernel times itself: kernels' clocks can differ by ~0.1 s *)
   While[AbsoluteTime[] < deadline,
    j++; res = lifetime50Run[seedBase + j, loss50C];
    If[AbsoluteTime[] < deadline, runs++;
     If[Last[Last[res]] === 0,
      AppendTo[sols, {AbsoluteTime[] - t0, Length[res] - 1, FromDigits[Flatten[res[[-1, 1]]] - 1, 2]}]]]];
   <|"runs_finished" -> runs, "solutions" -> sols|>];

kernels = ToExpression[Replace[Environment["WOLFRAM_KERNELS"], $Failed -> "8"]];
startParallel[] := If[Length[Kernels[]] == 0, LaunchKernels[kernels];
   DistributeDefinitions[RuleArray, MetaRuleArray, RuleArrayStates, RuleArrayLifetime, LifetimeLoss,
    LifetimeLossValue, IndicesToRules, MutateRuleArray, AdaptRuleArray, lifetime50Run, andXorRun, summary,
    compiledLoss, opts1, optsHalf, adaptLoop, harvestKernel, scaled];
   $DistributedContexts = None];         (* no automatic re-distribution: it would clear the subkernels' compiled losses *)
bits[rows_] := Map[ToExpression /@ Characters[#] &, rows];
data = Import[FileNameJoin[{root, "tests", "data", "wolfram_rule_arrays.json"}], "RawJSON"];

(* --------------------------------------------------------------------------------------------------------- verify *)
If[MemberQ[phases, "verify"],
  out["verify"] = Association @ Table[
    With[{panels = data["figures", fig, "panels"]},
     fig -> With[{checks = Table[
          With[{ra = bits[p["rule_array"]] /. {0 -> 4, 1 -> 146}, states = bits[p["states"]]},
           <|"lifetime" -> RuleArrayLifetime[opts1][First[states], ra, Length[ra]],
            "states_equal" -> (RuleArrayStates[opts1][First[states], ra, Length[ra]] === states),
            "cells" -> Length[Flatten[Rest[states]]]|>], {p, panels}]},
       <|"arrays" -> Length[checks], "cells" -> Total[Lookup[checks, "cells"]],
        "lifetimes" -> Lookup[checks, "lifetime"], "all_states_equal" -> And @@ Lookup[checks, "states_equal"]|>]],
    {fig, Keys[data["figures"]]}];
  KeyValueMap[Print["verify ", #1, ": ", #2["arrays"], " arrays, ", #2["cells"], " cells, states equal: ",
     #2["all_states_equal"], ", lifetimes ", Union[#2["lifetimes"]]] &, out["verify"]];
  save[];
  ];

(* --------------------------------------------------------------------------------------------------------- replay *)
If[MemberQ[phases, "replay"],
  startParallel[];
  t0 = AbsoluteTime[];
  evols = ParallelTable[summary[lifetime50Run[234234 + i]], {i, 1, 100}, Method -> "FinestGrained"];
  out["replay_lifetime50_seconds"] = AbsoluteTime[] - t0;
  out["replay_lifetime50"] = evols;
  (* figure 5 shows evols[[{1, 2, 3, 4, 5, 6, 7, 90, 97, 89, 20, 13}]] in a 6-column grid *)
  shown = {1, 2, 3, 4, 5, 6, 7, 90, 97, 89, 20, 13};
  published = #["rule_array"] & /@ data["figures", "lifetime50_solutions", "panels"];
  out["replay_figure5"] = <|"runs_shown" -> shown,
     "matches_published_in_order" -> Table[evols[[shown[[k]], "array"]] === published[[k]], {k, Length[shown]}],
     "published_found_among_100" -> Table[FirstPosition[evols[[All, "array"]], p, {Missing[]}][[1]], {p, published}]|>;
  out["replay_lifetime50_seed234241"] = summary[lifetime50Run[234241]];
  out["replay_andxor_seed234234"] = summary[andXorRun[234234]];
  save[];
  Print["replay: lifetime-50 seeds 234235..234334 in ", Round[out["replay_lifetime50_seconds"]], " s; solved ",
   Count[evols, _?(#["solved"] &)], "/100; figure-5 arrays reproduced: ",
   Count[out["replay_figure5", "matches_published_in_order"], True], "/12"];
  Print["replay: seed 234241: ", KeyDrop[out["replay_lifetime50_seed234241"], {"array", "loss_trajectory"}]];
  Print["replay: And/Xor seed 234234: ", KeyDrop[out["replay_andxor_seed234234"], {"array", "loss_trajectory"}]];
  ];

(* --------------------------------------------------------------------------------------------------------- detail *)
(* the And/Xor run of figure 20 mutation by mutation, for the essay's "1705 mutations, only 780 of them actually
   contribute to generating the final rule array": counts under several readings of "contribute" *)
If[MemberQ[phases, "detail"],
  res = andXorRun[234234];
  arrays = res[[All, 1]]; losses = res[[All, 2]];
  states = RuleArrayStates[optsHalf][CenterArray[1, 21], IndicesToRules[#, {8, 6}, optsHalf], 40] & /@ arrays;
  acc = Select[Range[Length[arrays] - 1], arrays[[#]] =!= arrays[[# + 1]] &];
  cellOf[k_] := First[SparseArray[arrays[[k + 1]] - arrays[[k]]]["NonzeroPositions"]];
  flips = Counts[cellOf /@ acc];
  out["detail_andxor_seed234234"] = <|"mutations" -> Length[res] - 1, "accepted" -> Length[acc],
     "accepted_changing_pattern" -> Count[acc, k_ /; states[[k]] =!= states[[k + 1]]],
     "accepted_changing_lifetime" -> Count[acc, k_ /; losses[[k]] =!= losses[[k + 1]]],
     "distinct_cells_flipped" -> Length[flips], "cells_flipped_odd_times" -> Count[flips, _?OddQ],
     "cells" -> 40*21,
     (* mutation k (1-based) produced res[[k + 1]]; for plotting the run as it evolves *)
     "pattern_changing_mutations" -> Select[acc, states[[#]] =!= states[[# + 1]] &],
     "lifetime_changes" -> ({#, ToString[losses[[# + 1]]]} & /@ Select[acc, losses[[#]] =!= losses[[# + 1]] &])|>;
  Print["detail: And/Xor seed 234234: ", out["detail_andxor_seed234234"]];
  save[];
  ];

(* --------------------------------------------------------------------------------------------------------- sample *)
If[MemberQ[phases, "sample"],
  startParallel[];
  n = ToExpression[Replace[Environment["WOLFRAM_SAMPLE"], $Failed -> "4096"]];
  short[res_] := {Length[res] - 1, Last[Last[res]] === 0};
  DistributeDefinitions[short];
  (* compiled on every subkernel; the timing phase checks that compiled and interpreted runs are identical *)
  ParallelEvaluate[loss50C = compiledLoss[50, 64, 31, opts1]; lossXorC = compiledLoss[30, 40, 21, optsHalf];];
  check = ParallelEvaluate[{lifetime50Run[234235, loss50C] === lifetime50Run[234235],
      andXorRun[234234, lossXorC][[-1]] === andXorRun[234234][[-1]]}, First[Kernels[]]];
  Print["sample: compiled runs identical to interpreted: ", check];
  t0 = AbsoluteTime[];
  s50 = ParallelTable[short[lifetime50Run[10^6 + i, loss50C]], {i, 1, n}, Method -> "FinestGrained"];
  t1 = AbsoluteTime[];
  sxor = ParallelTable[short[andXorRun[2 10^6 + i, lossXorC]], {i, 1, n}, Method -> "FinestGrained"];
  t2 = AbsoluteTime[];
  out["sample"] = <|"runs" -> n, "kernels" -> Length[Kernels[]], "loss" -> "compiled",
     "compiled_identical_to_interpreted" -> check,
     "lifetime50" -> <|"seeds" -> "10^6 + i", "mutations" -> s50[[All, 1]], "solved" -> s50[[All, 2]],
       "seconds" -> t1 - t0|>,
     "andxor" -> <|"seeds" -> "2 10^6 + i", "mutations" -> sxor[[All, 1]], "solved" -> sxor[[All, 2]],
       "seconds" -> t2 - t1|>|>;
  save[];
  Print["sample: ", n, " runs per setup on ", Length[Kernels[]], " kernels in ", Round[t2 - t0], " s"];
  ];

(* --------------------------------------------------------------------------------------------------------- timing *)
If[MemberQ[phases, "timing"],
  lossFn = LifetimeLoss[50][64, 31, opts1];
  published = (bits[#["rule_array"]] /. {0 -> 4, 1 -> 146}) & /@ data["figures", "lifetime50_solutions", "panels"];
  allFour = ConstantArray[4, {64, 31}];                    (* the starting array: the cell survives all 64 steps *)
  perCandidate[f_, ra_] := First[AbsoluteTiming[Do[f[ra], {50}]]]/50;
  interp = <|"candidate_seconds_full_64_steps" -> perCandidate[lossFn, allFour],
     "candidate_seconds_lifetime_50" -> Mean[perCandidate[lossFn, #] & /@ Take[published, 4]]|>;
  {tRun, res} = AbsoluteTiming[lifetime50Run[234235]];
  interp["run_seconds"] = tRun;
  interp["run_mutations"] = Length[res] - 1;
  interp["candidates_per_second"] = (Length[res] - 1)/tRun;
  out["timing_interpreted"] = interp;
  Print["timing, interpreted: ", interp];
  save[];
  (* the same RuleArray function compiled; MetaRuleArray already specialises it for FunctionCompile *)
  {tCompile, compiled} = AbsoluteTiming[Quiet[Check[FunctionCompile[MetaRuleArray[opts1, "Function" -> "Lifetime"]],
       $Failed]]];
  If[Head[compiled] === CompiledCodeFunction,
   lossC = Function[ra, LifetimeLossValue[compiled[CenterArray[1, 31], ra, 64], 50, 65]];
   same = And @@ Table[lossC[r] === lossFn[r], {r, Join[published, {allFour}]}];
   {tRunC, resC} = AbsoluteTiming[lifetime50Run[234235, lossC]];
   out["timing_compiled"] = <|"compile_seconds" -> tCompile, "losses_equal_interpreted" -> same,
      "run_identical_to_interpreted" -> (resC === res),
      "candidate_seconds_full_64_steps" -> perCandidate[lossC, allFour],
      "candidate_seconds_lifetime_50" -> Mean[perCandidate[lossC, #] & /@ Take[published, 4]],
      "run_seconds" -> tRunC, "candidates_per_second" -> (Length[resC] - 1)/tRunC|>;
   Print["timing, compiled: ", out["timing_compiled"]],
   out["timing_compiled"] = <|"error" -> ToString[compiled, InputForm]|>;
   Print["FunctionCompile failed: ", compiled]];
  out["timing_kernels"] = 1;
  out["timing_cpu"] = $ProcessorType <> ", " <> ToString[$ProcessorCount] <> " cores";
  save[];
  ];

(* ============================================================ equal-time phases (see benchmarks/equal_time.py) *)
nullMin[l_List] := If[l === {}, Null, Min[l]];
kernelInfo[] := <|"version" -> $Version, "cpu" -> $ProcessorType, "processors" -> $ProcessorCount,
   "machine" -> $MachineName, "slurm_job" -> Replace[Environment["SLURM_JOB_ID"], $Failed -> Null]|>;

(* ------------------------------------------------------------------------------------------------------ loopcheck *)
If[MemberQ[phases, "loopcheck"],
  lossCheck = compiledLoss[50, 64, 31, opts1];
  ref = (SeedRandom[234235]; AdaptRuleArray[ConstantArray[1, {64, 31}], {4, 146}, 10000, lossCheck, opts1]);
  mine = (SeedRandom[234235];
    adaptLoop[ConstantArray[1, {64, 31}], {4, 146}, lossCheck, Infinity, AbsoluteTime[], False, 10000, opts1]);
  out["loopcheck"] = <|"seed" -> 234235, "same_final_array" -> (mine["final"] === ref[[-1, 1]]),
     "same_mutations" -> (mine["mutations"] == Length[ref] - 1), "mutations" -> mine["mutations"]|>;
  Print["loopcheck: ", out["loopcheck"]];
  save[];
  ];

(* -------------------------------------------------------------------------------------------------------- latency *)
(* one run at a time on one kernel, his code as published and with RuleArray compiled *)
If[MemberQ[phases, "latency"],
  lossLat = compiledLoss[50, 64, 31, opts1];
  oneRun[seed_, lf_] := Module[{t, res}, {t, res} = AbsoluteTiming[lifetime50Run[seed, lf]];
     <|"seconds" -> t, "mutations" -> Length[res] - 1, "solved" -> (Last[Last[res]] === 0)|>];
  oneRun[1, lossLat];
  out["latency"] = <|"kernels" -> 1, "host" -> kernelInfo[],
     "interpreted" -> Table[oneRun[5 10^6 + i, Automatic], {i, envNum["WOLFRAM_LATENCY_INTERP", 20]}],
     "compiled" -> Table[oneRun[6 10^6 + i, lossLat], {i, envNum["WOLFRAM_LATENCY_COMPILED", 200]}]|>;
  Print["latency: median ", Median[Lookup[out["latency", "interpreted"], "seconds"]], " s interpreted, ",
   Median[Lookup[out["latency", "compiled"], "seconds"]], " s compiled"];
  save[];
  ];

(* ---------------------------------------------------------------------------------------------------------- first *)
(* one run per subkernel started together (compiled loss): seconds until the first solution reaches the master *)
If[MemberQ[phases, "first"],
  startParallel[];
  ParallelEvaluate[loss50C = compiledLoss[50, 64, 31, opts1];];
  firstRound[r_] := Module[{t0 = AbsoluteTime[], ends},
     ends = ParallelTable[Module[{tk = AbsoluteTime[], res},     (* durations on each kernel's own clock *)
        res = lifetime50Run[7 10^6 + 100 r + i, loss50C];
        {AbsoluteTime[] - tk, Last[Last[res]] === 0, Length[res] - 1}], {i, 1, Length[Kernels[]]},
       Method -> "FinestGrained"];
     <|"seconds" -> nullMin[Cases[ends, {t_, True, _} :> t]], "round_seconds_on_master" -> AbsoluteTime[] - t0,
      "runs" -> ends|>];
  firstRound[0];
  out["first"] = <|"kernels" -> Length[Kernels[]], "loss" -> "compiled", "host" -> kernelInfo[],
     "rounds" -> Table[firstRound[r], {r, envNum["WOLFRAM_FIRST_REPEATS", 50]}]|>;
  Print["first: median ", Median[DeleteCases[Lookup[out["first", "rounds"], "seconds"], Null]], " s"];
  save[];
  ];

(* -------------------------------------------------------------------------------------------------------- harvest *)
(* lifetime-50 runs on every subkernel for a fixed budget (compiled loss): every solution, with hashes of its rule
   array and of its pattern, for distinct-solution counts *)
If[MemberQ[phases, "harvest"],
  startParallel[];
  tCompile = First[AbsoluteTiming[ParallelEvaluate[loss50C = compiledLoss[50, 64, 31, opts1];]]];
  budget = envNum["WOLFRAM_BUDGET", 3600];
  t0 = AbsoluteTime[];
  hv = With[{b = budget}, ParallelEvaluate[harvestKernel[b, 10^7 $KernelID]]];
  tEnd = AbsoluteTime[] - t0;
  toArray[n_] := Partition[IntegerDigits[n, 2, 64 31], 31];
  sols = Flatten[Table[Append[#, k] & /@ hv[[k, "solutions"]], {k, Length[hv]}], 1];
  out["harvest"] = <|"kernels" -> Length[Kernels[]], "budget" -> budget, "seconds" -> tEnd,
     "compile_seconds" -> tCompile, "host" -> kernelInfo[], "runs_finished" -> Total[Lookup[hv, "runs_finished"]],
     "columns" -> {"seconds", "mutations", "hash_array", "hash_pattern", "rule146_cells", "kernel"},
     "solutions" -> ({#[[1]], #[[2]], Hash[toArray[#[[3]]], "SHA256", "HexString"],
          Hash[RuleArrayStates[opts1][CenterArray[1, 31], toArray[#[[3]]] /. {0 -> 4, 1 -> 146}, 64], "SHA256",
           "HexString"], DigitCount[#[[3]], 2, 1], #[[4]]} & /@ SortBy[sols, First])|>;
  Print["harvest: ", Length[sols], " solutions in ", Round[tEnd], " s on ", Length[Kernels[]], " kernels"];
  save[];
  ];

(* ---------------------------------------------------------------------------------------------- anytime and reach *)
(* runs restarted after each solution for a fixed budget at a scaled size; "interpreted": his code as published on
   one kernel; "compiled": RuleArray compiled, one run per subkernel *)
budgetRun[sz_, budget_, mode_, seed_] := Module[{t0, res, tc = 0},
   If[mode == "interpreted",
    SeedRandom[seed];
    t0 = AbsoluteTime[];
    res = {KeyDrop[adaptLoop[ConstantArray[1, {sz["height"], sz["width"]}], {4, 146},
        LifetimeLoss[sz["target"]][sz["height"], sz["width"], opts1], t0 + budget, t0, True, Infinity, opts1],
       "final"]},
    startParallel[];
    tc = First[AbsoluteTiming[With[{tg = sz["target"], h = sz["height"], w = sz["width"]},
        ParallelEvaluate[lossS = compiledLoss[tg, h, w, opts1];]]]];
    t0 = AbsoluteTime[];
    res = With[{b = budget, h = sz["height"], w = sz["width"], sd = seed},     (* each kernel times itself *)
      ParallelEvaluate[SeedRandom[sd + $KernelID]; Module[{tk = AbsoluteTime[]},
        KeyDrop[adaptLoop[ConstantArray[1, {h, w}], {4, 146}, lossS, tk + b, tk, True, Infinity, opts1], "final"]]]]];
   <|"setup" -> sz, "mode" -> mode, "kernels" -> Length[res], "budget" -> budget,
    "seconds" -> AbsoluteTime[] - t0, "compile_seconds" -> tc, "host" -> kernelInfo[],
    "mutations" -> Total[Lookup[res, "mutations"]],
    "solutions" -> Total[Length /@ Lookup[res, "solutions"]],
    "first_solution_s" -> nullMin[Flatten[#[[All, 1]] & /@ Lookup[res, "solutions"]]],
    "per_kernel" -> res|>];
If[MemberQ[phases, "anytime"],
  mode = Replace[Environment["WOLFRAM_MODE"], $Failed -> "compiled"];
  r = budgetRun[scaled[envNum["WOLFRAM_SCALE", 10]], envNum["WOLFRAM_BUDGET", 3600], mode, 8 10^6];
  out["anytime_" <> mode] = r;
  Print["anytime (", mode, "): ", r["solutions"], " solutions, ", r["mutations"], " mutations in ",
   Round[r["seconds"]], " s; first solution at ", r["first_solution_s"]];
  save[];
  ];
If[MemberQ[phases, "reach"],
  mode = Replace[Environment["WOLFRAM_MODE"], $Failed -> "compiled"];
  key = "reach_" <> mode;
  out[key] = {};
  Do[r = budgetRun[scaled[s], envNum["WOLFRAM_REACH_BUDGET", 600], mode, 9 10^6 + 1000 s];
   out[key] = Append[out[key], r];
   Print["reach (", mode, ") scale ", s, ": ", r["solutions"], " solutions, ", r["mutations"], " mutations, first at ",
    r["first_solution_s"]];
   save[],
   {s, ToExpression /@ StringSplit[Replace[Environment["WOLFRAM_SCALES"], $Failed -> "1,2,4,6,8,10,14,20"], ","]}];
  ];

(* ------------------------------------------------------------------------------------------------- ladder_timing *)
(* seconds per mutation (his loop: mutate, IndicesToRules, loss) from the starting array, whose pattern survives every
   step, at each scaled size; interpreted and compiled, one kernel *)
If[MemberQ[phases, "ladder_timing"],
  out["ladder_timing"] = Table[With[{sz = scaled[s]},
     Module[{ra = ConstantArray[1, {sz["height"], sz["width"]}], lf, lc, per},
      lf = LifetimeLoss[sz["target"]][sz["height"], sz["width"], opts1];
      lc = compiledLoss[sz["target"], sz["height"], sz["width"], opts1];
      per[f_] := Module[{n = 1, t},
        While[(t = First[AbsoluteTiming[adaptLoop[ra, {4, 146}, f, Infinity, 0, False, n, opts1]]]) < 2 && n < 4096,
         n *= 4];
        t/n];
      <|"setup" -> sz, "interpreted_seconds_per_mutation" -> per[lf],
       "compiled_seconds_per_mutation" -> per[lc]|>]],
    {s, ToExpression /@ StringSplit[Replace[Environment["WOLFRAM_SCALES"], $Failed -> "1,2,4,6,8,10,14,20"], ","]}];
  out["ladder_timing_host"] = kernelInfo[];
  Print["ladder_timing: ", {#["setup", "scale"], #["interpreted_seconds_per_mutation"],
      #["compiled_seconds_per_mutation"]} & /@ out["ladder_timing"]];
  save[];
  ];

Print["wrote ", outFile];
