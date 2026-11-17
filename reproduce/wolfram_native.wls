#!/usr/bin/env wolframscript
(* Wolfram's lifetime experiments run with Wolfram's own code, in the Wolfram Language.

     wolframscript -file reproduce/wolfram_native.wls [phase ...]

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

   His code is not copied into this repository: it is loaded from the "copy code" file attached to figure 3 of the essay
   and only its definitions are evaluated (no plotting, no cloud calls). The file holds notebook boxes; each input cell is
   turned back into linear code by joining its tokens and parsed with ToExpression. Results go to
   reproduce/results/wolfram_native.json; reproduce/compare_wolfram.py compares them with STICA.
   Parallel phases use subkernels (environment variable WOLFRAM_KERNELS, default 8); run under a memory cap, e.g.
     systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 wolframscript -file reproduce/wolfram_native.wls *)

root = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
phases = If[Length[$ScriptCommandLine] > 1, Rest[$ScriptCommandLine], {"verify", "replay", "detail", "timing"}];
outFile = FileNameJoin[{root, "reproduce", "results", "wolfram_native.json"}];
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
codeText = Import[codeURL, "Text"];
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

kernels = ToExpression[Replace[Environment["WOLFRAM_KERNELS"], $Failed -> "8"]];
startParallel[] := If[Length[Kernels[]] == 0, LaunchKernels[kernels];
   DistributeDefinitions[RuleArray, MetaRuleArray, RuleArrayStates, RuleArrayLifetime, LifetimeLoss,
    LifetimeLossValue, IndicesToRules, MutateRuleArray, AdaptRuleArray, lifetime50Run, andXorRun, summary,
    compiledLoss, opts1, optsHalf];
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
     "cells" -> 40*21|>;
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

Print["wrote reproduce/results/wolfram_native.json"];
