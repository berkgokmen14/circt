//===- AffineToStaticlogic.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToSTG.h"
#include "../PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/Pipeline/Pipeline.h"
#include "circt/Dialect/STG/STG.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scf-to-stg"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::pipeline;
using namespace circt::stg;

namespace {

struct AffineToSTG : public AffineToSTGBase<AffineToSTG> {
  void runOnOperation() override;

private:
  LogicalResult
  lowerAffineStructures(MemoryDependenceAnalysis &dependenceAnalysis);
  LogicalResult populateOperatorTypes(WhileOp &whileOp);
  LogicalResult solveSchedulingProblem(WhileOp &whileOp);
  LogicalResult createSTGSTG(WhileOp &whileOp);

  SharedOperatorsSchedulingAnalysis *schedulingAnalysis;
};

} // namespace

void AffineToSTG::runOnOperation() {
  // Get dependence analysis for the whole function.
  auto dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  OpBuilder builder(getOperation());

  // Attach trip count for affine for loops
  // getOperation().walk([&](AffineForOp loop) {
  //   auto tripCount = getConstantTripCount(loop);
  //   if (tripCount.has_value()) {
  //     auto attr = builder.getI64IntegerAttr(tripCount.value());
  //     loop->setAttr("stg.tripCount", attr);
  //   }
  //   return WalkResult::advance();
  // });
  // After dependence analysis, materialize affine structures.
  if (failed(lowerAffineStructures(dependenceAnalysis)))
    return signalPassFailure();

  // getOperation().dump();

  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<SharedOperatorsSchedulingAnalysis>();

  SmallVector<WhileOp> loops;

  getOperation().walk([&](WhileOp loop) {
    loops.push_back(loop);
    return WalkResult::advance();
  });

  for (auto loop : loops) {
    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(loop)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createSTGSTG(loop)))
      return signalPassFailure();
  }

  // getOperation().dump();
}

struct ForLoopLoweringPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Generate type signature for the loop-carried values. The induction
    // variable is placed first, followed by the forOp.iterArgs.
    SmallVector<Type> lcvTypes;
    SmallVector<Location> lcvLocs;
    lcvTypes.push_back(forOp.getInductionVar().getType());
    lcvLocs.push_back(forOp.getInductionVar().getLoc());
    for (Value value : forOp.getInitArgs()) {
      lcvTypes.push_back(value.getType());
      lcvLocs.push_back(value.getLoc());
    }

    // Build scf.WhileOp
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());
    auto whileOp = rewriter.create<WhileOp>(forOp.getLoc(), lcvTypes, initArgs,
                                            forOp->getAttrs());

    // 'before' region contains the loop condition and forwarding of iteration
    // arguments to the 'after' region.
    auto *beforeBlock = rewriter.createBlock(
        &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
    rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
    auto cmpOp = rewriter.create<arith::CmpIOp>(
        whileOp.getLoc(), arith::CmpIPredicate::slt,
        beforeBlock->getArgument(0), forOp.getUpperBound());
    rewriter.create<scf::ConditionOp>(whileOp.getLoc(), cmpOp.getResult(),
                                      beforeBlock->getArguments());

    // Inline for-loop body into an executeRegion operation in the "after"
    // region. The return type of the execRegionOp does not contain the
    // iv - yields in the source for-loop contain only iterArgs.
    auto *afterBlock = rewriter.createBlock(
        &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);

    // Add induction variable incrementation
    rewriter.setInsertionPointToEnd(afterBlock);
    auto ivIncOp = rewriter.create<arith::AddIOp>(
        whileOp.getLoc(), afterBlock->getArgument(0), forOp.getStep());

    // Rewrite uses of the for-loop block arguments to the new while-loop
    // "after" arguments
    for (const auto &barg : enumerate(forOp.getBody(0)->getArguments()))
      barg.value().replaceAllUsesWith(afterBlock->getArgument(barg.index()));

    // Inline for-loop body operations into 'after' region.
    for (auto &arg : llvm::make_early_inc_range(*forOp.getBody()))
      arg.moveBefore(afterBlock, afterBlock->end());

    // Add incremented IV to yield operations
    for (auto yieldOp : afterBlock->getOps<scf::YieldOp>()) {
      SmallVector<Value> yieldOperands = yieldOp.getOperands();
      yieldOperands.insert(yieldOperands.begin(), ivIncOp.getResult());
      yieldOp->setOperands(yieldOperands);
    }

    // We cannot do a direct replacement of the forOp since the while op returns
    // an extra value (the induction variable escapes the loop through being
    // carried in the set of iterargs). Instead, rewrite uses of the forOp
    // results.
    for (const auto &arg : llvm::enumerate(forOp.getResults()))
      arg.value().replaceAllUsesWith(whileOp.getResult(arg.index() + 1));

    rewriter.eraseOp(forOp);
    return success();
  }
};

/// Helper to hoist computation out of scf::IfOp branches, turning it into a
/// mux-like operation, and exposing potentially concurrent execution of its
/// branches.
struct IfOpHoisting : OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.mergeBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.mergeBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

/// Helper to determine if an scf::IfOp is in mux-like form.
static bool ifOpLegalityCallback(IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

/// Helper to mark AffineYieldOp legal, unless it is inside a partially
/// converted scf::IfOp.
static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<IfOp>();
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially complex
/// computations in the condition of ifs, or the addresses of loads and stores.
/// The dependence analysis will be updated so the dependences from the affine
/// loads and stores are now on the memref loads and stores.
LogicalResult AffineToSTG::lowerAffineStructures(
    MemoryDependenceAnalysis &dependenceAnalysis) {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect>();
  target.addIllegalOp<AffineIfOp, mlir::AffineLoadOp, mlir::AffineStoreOp,
                      AffineApplyOp, AffineForOp, AffineYieldOp>();
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  // target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  // patterns.add<AffineLoadLowering>(context, dependenceAnalysis);
  // patterns.add<AffineStoreLowering>(context, dependenceAnalysis);
  patterns.add<IfOpHoisting>(context);
  patterns.add<ForLoopLoweringPattern>(context);
  populateAffineToStdConversionPatterns(patterns);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Loop invariant code motion to hoist produced constants out of loop
  op->walk(
     [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

  patterns.clear();
  patterns.add<ForLoopLoweringPattern>(context);
  target.addIllegalOp<scf::ForOp>();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();


  return success();
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult AffineToSTG::populateOperatorTypes(
    WhileOp &whileOp) {
  // Scheduling analyis only considers the innermost loop nest for now.
  // auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(whileOp);

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Operation *unsupported;
  WalkResult result = whileOp.getAfter().walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AffineYieldOp, arith::ConstantOp, IndexCastOp, 
              memref::AllocaOp, YieldOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<memref::LoadOp, memref::StoreOp, IfOp, AddIOp, CmpIOp>(
            [&](Operation *seqOp) {
              // Some known sequential ops. In certain cases, reads may be
              // combinational in Calyx, but taking advantage of that is left as
              // a future enhancement.
              problem.setLinkedOperatorType(seqOp, seqOpr);
              return WalkResult::advance();
            })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return whileOp.emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToSTG::solveSchedulingProblem(
    WhileOp &whileOp) {
  // Scheduling analyis only considers the innermost loop nest for now.
  // auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(whileOp);

  // Optionally debug problem inputs.
  LLVM_DEBUG(
  whileOp.getAfter().walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { "
                     << "source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = whileOp.getAfter().getBlocks().back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG(
  whileOp.getAfter().walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling outputs for " << *op;
    llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
    llvm::dbgs() << "\n\n";
  }));
  
  return success();
}


DenseMap<int64_t, SmallVector<Operation*>> getOperationCycleMap(Problem &problem) {
  DenseMap<int64_t, SmallVector<Operation*>> map;

  for (auto *op : problem.getOperations()) {
    auto cycleOpt = problem.getStartTime(op);
    assert(cycleOpt.has_value());
    auto cycle = cycleOpt.value();
    auto vec = map.lookup(cycle);
    vec.push_back(op);
    map.insert(std::pair(cycle, vec));
  }

  return map;
}

int64_t longestOperationStartingAtTime(Problem &problem, const DenseMap<int64_t, SmallVector<Operation*>>& opMap, int64_t cycle) {
  int64_t longestOp = 0;
  for (auto *op : opMap.lookup(cycle)) {
    auto oprType = problem.getLinkedOperatorType(op);
    assert(oprType.has_value());
    auto latency = problem.getLatency(oprType.value());
    assert(latency.has_value());
    if (latency.value() > longestOp) 
      longestOp = latency.value();
  }

  return longestOp;
}

/// Create the pipeline op for a loop nest.
LogicalResult AffineToSTG::createSTGSTG(
    WhileOp &whileOp) {
  auto anchor = whileOp.getYieldOp();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(whileOp);

  auto opMap = getOperationCycleMap(problem);

  // auto outerLoop = loopNest.front();
  // auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(whileOp.getLoc(), whileOp);

  // // Create Values for the loop's lower and upper bounds.
  // Value lowerBound = lowerAffineLowerBound(innerLoop, builder);
  // Value upperBound = lowerAffineUpperBound(innerLoop, builder);
  // int64_t stepValue = innerLoop.getStep();
  // auto step = builder.create<arith::ConstantOp>(
  //     IntegerAttr::get(builder.getIndexType(), stepValue));

  // // Create the pipeline op, with the same result types as the inner loop. An
  // // iter arg is created for the induction variable.
  // TypeRange resultTypes = innerLoop.getResultTypes();

  // auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  // SmallVector<Value> iterArgs;
  // iterArgs.push_back(lowerBound);
  // iterArgs.append(innerLoop.getIterOperands().begin(),
  //                 innerLoop.getIterOperands().end());
  auto iterArgs = whileOp.getInits();

  SmallVector<Type> resultTypes;

  for (auto value : whileOp.getAfterArguments()) {
    if (value.isUsedOutsideOfBlock(&whileOp.getAfter().front()))
        resultTypes.push_back(value.getType());
  }

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  // Optional<IntegerAttr> tripCountAttr;
  // if (whileOp->hasAttr("stg.tripCount")) {
  //   tripCountAttr = whileOp->getAttr("stg.tripCount").cast<IntegerAttr>();
  // }

  auto condValue = builder.getIntegerAttr(builder.getIndexType(), 1);
  auto cond = builder.create<arith::ConstantOp>(whileOp.getLoc(), condValue);

  auto stgWhile = builder.create<stg::STGWhileOp>(whileOp.getLoc(), resultTypes, llvm::None, iterArgs);

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  BlockAndValueMapping valueMap;
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(whileOp.getBefore().getArgument(i),
                 stgWhile.getCondBlock().getArgument(i));

  builder.setInsertionPointToStart(&stgWhile.getCondBlock());

  // auto condConst = builder.create<arith::ConstantOp>(whileOp.getLoc(), builder.getIntegerAttr(builder.getI1Type(), 1));
  auto *conditionReg = stgWhile.getCondBlock().getTerminator();
  // conditionReg->insertOperands(0, condConst.getResult());
  for (auto &op : whileOp.getBefore().front().getOperations()) {
    if (isa<scf::ConditionOp>(op)) {
      auto condOp = cast<scf::ConditionOp>(op);
      auto cond = condOp.getCondition();
      auto condNew = valueMap.lookupOrNull(cond);
      assert(condNew);
      conditionReg->insertOperands(0, condNew);     
    } else {
      auto *newOp = builder.clone(op, valueMap);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = newOp->getResult(i);
        auto oldValue = op.getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }
  }

  builder.setInsertionPointToStart(&stgWhile.getScheduleBlock());

  // auto termConst = builder.create<arith::ConstantOp>(whileOp.getLoc(), builder.getIndexAttr(1));
  auto term = stgWhile.getTerminator();
  // term.getIterArgsMutable().append(termConst.getResult());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  SmallVector<SmallVector<Operation*>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  valueMap.clear();
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(whileOp.getAfter().getArgument(i),
                 stgWhile.getScheduleBlock().getArgument(i));

  // Create the stages.
  Block &scheduleBlock = stgWhile.getScheduleBlock();
  builder.setInsertionPointToStart(&scheduleBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto& group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    OpBuilder::InsertionGuard g(builder);

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [whileOp](Operation *op) {
      return isa<YieldOp>(op) && op->getParentOp() == whileOp;
    };
    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      for (auto *user : op->getUsers()) {
        if (*problem.getStartTime(user) > startTime || isLoopTerminator(user)) {
          if (!opsWithReturns.contains(op)) {
            opsWithReturns.insert(op);
            stepTypes.append(op->getResultTypes().begin(),
                              op->getResultTypes().end());
          }
        }
      }
    }

    // Create the stage itself.
    auto startTimeAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), startTime);
    auto stage =
        builder.create<STGStepOp>(stepTypes, startTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    // Sort the group according to original dominance.
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });

    // Move over the operations and add their results to the terminator.
    SmallVector<std::tuple<Operation *, Operation *, unsigned>> movedOps;
    for (auto *op : group) {
      unsigned resultIndex = stageTerminator->getNumOperands();
      auto *newOp = builder.clone(*op, valueMap);
      if (opsWithReturns.contains(op)) {
        stageTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
    }

    // Add the stage results to the value map for the original op.
    for (auto tuple : movedOps) {
      Operation *op = std::get<0>(tuple);
      Operation *newOp = std::get<1>(tuple);
      unsigned resultIndex = std::get<2>(tuple);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = stage->getResult(resultIndex + i);
        auto oldValue = op->getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }
  }

  // Add the iter args and results to the terminator.
  auto scheduleTerminator =
      cast<STGTerminatorOp>(scheduleBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  // termIterArgs.push_back(
  //     scheduleBlock.front().getResult(scheduleBlock.front().getNumResults() - 1));
  for (auto value : whileOp.getYieldOp()->getOperands()) {
    termIterArgs.push_back(valueMap.lookup(value));
    if (value.isUsedOutsideOfBlock(&whileOp.getAfter().front()))
      termResults.push_back(valueMap.lookup(value));
  }

  scheduleTerminator.getIterArgsMutable().append(termIterArgs);
  scheduleTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < whileOp.getNumResults(); ++i)
    whileOp.getResult(i).replaceAllUsesWith(stgWhile.getResult(i));

  // Remove the loop nest from the IR.
  whileOp.walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass> circt::createAffineToSTGPass() {
  return std::make_unique<AffineToSTG>();
}
