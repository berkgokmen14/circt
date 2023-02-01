//===- AffineToSTG.cpp ----------------------------------------------------===//
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
#include "mlir/IR/Visitors.h"
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
  LogicalResult populateOperatorTypes(Operation *op);
  LogicalResult solveSchedulingProblem(Operation *op);
  LogicalResult createWhileOpSTG(WhileOp &whileOp);
  LogicalResult createFuncOpSTG(FuncOp &funcOp);

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

  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<SharedOperatorsSchedulingAnalysis>();

  SmallVector<WhileOp> loops;

  getOperation().walk([&](WhileOp loop) {
    loops.push_back(loop);
    return WalkResult::advance();
  });

  // Schedule loops
  for (auto loop : loops) {
    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(loop)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createWhileOpSTG(loop)))
      return signalPassFailure();
  }

  // Schedule whole function
  auto funcOp = cast<FuncOp>(getOperation());

  // Populate the target operator types.
  if (failed(populateOperatorTypes(funcOp)))
    return signalPassFailure();

  // Solve the scheduling problem computed by the analysis.
  if (failed(solveSchedulingProblem(funcOp)))
    return signalPassFailure();

  // Convert the IR.
  if (failed(createFuncOpSTG(funcOp)))
    return signalPassFailure();
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
    Operation *op) {
  // Scheduling analyis only considers the innermost loop nest for now.
  // auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(op);

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Region *region;
  if (auto whileOp = dyn_cast<WhileOp>(op); whileOp)
    region = &whileOp.getAfter();
  else if (auto funcOp = dyn_cast<FuncOp>(op); funcOp)
    region = &funcOp.getBody();
  else {
    op->emitOpError("Unsupported operation for operator type population");
    return failure();
  }

  DenseMap<Value, int64_t> memrefs;

  Operation *unsupported;
  WalkResult result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getParentOfType<STGWhileOp>() != nullptr
        || op->getParentOfType<PipelineWhileOp>() != nullptr) {
      return WalkResult::advance();
    }

    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AffineYieldOp, arith::ConstantOp, IndexCastOp, 
              memref::AllocaOp, YieldOp, ConditionOp,
              memref::AllocOp, func::ReturnOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<IfOp, AddIOp, SubIOp, CmpIOp, STGWhileOp>(
        [&](Operation *seqOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          problem.setLinkedOperatorType(seqOp, seqOpr);
          return WalkResult::advance();
        })
        .Case<memref::LoadOp, memref::StoreOp>([&](Operation *memOp) {
          // Handle resource constraints for memory ops
          Value memref;
          if (isa<memref::LoadOp>(*memOp)) {
            memref = cast<memref::LoadOp>(*memOp).getMemRef();
          } else {
            memref = cast<memref::StoreOp>(*memOp).getMemRef();
          }

          Problem::OperatorType memOpr;
          if (memrefs.count(memref) > 0) {
            auto name = "memory_" + std::to_string(memrefs.lookup(memref));
            memOpr = problem.getOrInsertOperatorType(name);
          } else {
            auto curr = memrefs.size();
            memrefs.insert(std::pair(memref, curr));
            auto name = "memory_" + std::to_string(curr);
            memOpr = problem.getOrInsertOperatorType(name);
            problem.setLatency(memOpr, 1);
            problem.setLimit(memOpr, 1);
          }

          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Case<PipelineWhileOp>([&](Operation *pipelineOp) {
          // Problem::OperatorType whileOpr = problem.getOrInsertOperatorType("while");
          // problem.setLatency(whileOpr, 3);
          problem.setLinkedOperatorType(pipelineOp, seqOpr);
          return WalkResult::advance();
        })
        // .Case<STGWhileOp>([&](Operation *whileOp) {
        //   Problem::OperatorType whileOpr = problem.getOrInsertOperatorType("while");
        //   problem.setLatency(whileOpr, 3);
        //   problem.setLinkedOperatorType(whileOp, whileOpr);
        //   return WalkResult::advance();
        // })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return op->emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToSTG::solveSchedulingProblem(
    Operation *op) {
  // Scheduling analyis only considers the innermost loop nest for now.
  // auto forOp = loopNest.back();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(op);

  Region *region;
  if (auto whileOp = dyn_cast<WhileOp>(op); whileOp)
    region = &whileOp.getAfter();
  else if (auto funcOp = dyn_cast<FuncOp>(op); funcOp)
    region = &funcOp.getBody();
  else {
    op->emitOpError("Unsupported operation for operator type population");
    return failure();
  }

  // op->dump();
  // Optionally debug problem inputs.
  // LLVM_DEBUG(
  // region->walk<WalkOrder::PreOrder>([&](Operation *op) {
  //   if (op->getParentOfType<STGWhileOp>() != nullptr) {
  //     return WalkResult::advance();
  //   }

  //   llvm::dbgs() << "Scheduling inputs for " << *op;
  //   auto opr = problem.getLinkedOperatorType(op);
  //   llvm::dbgs() << "\n  opr = " << opr;
  //   llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
  //   for (auto dep : problem.getDependences(op))
  //     if (dep.isAuxiliary()) {
  //       llvm::dbgs() << "\n  dep = { "
  //                    << "source = " << *dep.getSource() << " }";
  //     }
  //   llvm::dbgs() << "\n\n";
  //   return WalkResult::advance();
  //   });
  // }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = region->getBlocks().back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  // LLVM_DEBUG(
  // region->walk<WalkOrder::PreOrder>([&](Operation *op) {
  //   if (op->getParentOfType<STGWhileOp>() != nullptr)
  //     return;

  //   llvm::dbgs() << "Scheduling outputs for " << *op;
  //   llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
  //   llvm::dbgs() << "\n\n";
  // });
  
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

/// Returns true if the value is used outside of the given loop.
bool isUsedOutsideOfRegion(Value val, Block *block) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    Operation *u = user;
    while (!isa<ModuleOp>(u->getParentRegion()->getParentOp())) {
      if (u->getBlock() == block) {
        return false;
      }
      u = u->getParentRegion()->getParentOp();
    }
    return true;
  });
}

/// Create the stg ops for a loop nest.
LogicalResult AffineToSTG::createWhileOpSTG(
    WhileOp &whileOp) {
  auto anchor = whileOp.getYieldOp();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(whileOp);

  auto opMap = getOperationCycleMap(problem);

  ImplicitLocOpBuilder builder(whileOp.getLoc(), whileOp);

  // Get iter args
  auto iterArgs = whileOp.getInits();

  SmallVector<Type> resultTypes;

  for (size_t i = 0; i < whileOp.getNumResults(); ++i) {
    auto result = whileOp.getResult(i);
    auto numUses = std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      resultTypes.push_back(result.getType());
    }
  }

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  // Optional<IntegerAttr> tripCountAttr;
  // if (whileOp->hasAttr("stg.tripCount")) {
  //   tripCountAttr = whileOp->getAttr("stg.tripCount").cast<IntegerAttr>();
  // }

  // auto condValue = builder.getIntegerAttr(builder.getIndexType(), 1);
  // auto cond = builder.create<arith::ConstantOp>(whileOp.getLoc(), condValue);

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
    auto stage =
        builder.create<STGStepOp>(stepTypes);
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
  for (int i = 0, vals = whileOp.getYieldOp()->getNumOperands(); i < vals; ++i) {
    auto value = whileOp.getYieldOp().getOperand(i);
    auto result = whileOp.getResult(i);
    termIterArgs.push_back(valueMap.lookup(value));
    auto numUses = std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      termResults.push_back(valueMap.lookup(value));
    }
  }

  scheduleTerminator.getIterArgsMutable().append(termIterArgs);
  scheduleTerminator.getResultsMutable().append(termResults);


  // Replace loop results with while results.
  auto resultNum = 0;
  for (size_t i = 0; i < whileOp.getNumResults(); ++i) {
    auto result = whileOp.getResult(i);
    auto numUses = std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      whileOp.getResult(i).replaceAllUsesWith(stgWhile.getResult(resultNum++));
    }
  }

  // Remove the loop nest from the IR.
  whileOp.walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

int64_t opOrParentStartTime(Problem &problem, Operation *op) {
  Operation *currentOp = op;

  while (!isa<func::FuncOp>(currentOp)) {
    if (problem.hasOperation(currentOp)) {
      return problem.getStartTime(currentOp).value();
    }
    currentOp = currentOp->getParentOp();
  }
  op->emitOpError("Operation or parent does not have start time");
  return -1;
}

/// Create the stg ops for a loop nest.
LogicalResult AffineToSTG::createFuncOpSTG(
    FuncOp &funcOp) {
  auto *anchor = funcOp.getBody().back().getTerminator();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(funcOp);

  auto opMap = getOperationCycleMap(problem);

  // auto outerLoop = loopNest.front();
  // auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(funcOp.getLoc(), funcOp);

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  BlockAndValueMapping valueMap;
  // for (size_t i = 0; i < iterArgs.size(); ++i)
  //   valueMap.map(whileOp.getBefore().getArgument(i),
  //                stgWhile.getCondBlock().getArgument(i));

  builder.setInsertionPointToStart(&funcOp.getBody().front());

  // auto condConst = builder.create<arith::ConstantOp>(whileOp.getLoc(), builder.getIntegerAttr(builder.getI1Type(), 1));
  // auto *conditionReg = stgWhile.getCondBlock().getTerminator();
  // conditionReg->insertOperands(0, condConst.getResult());
  // for (auto &op : whileOp.getBefore().front().getOperations()) {
  //   if (isa<scf::ConditionOp>(op)) {
  //     auto condOp = cast<scf::ConditionOp>(op);
  //     auto cond = condOp.getCondition();
  //     auto condNew = valueMap.lookupOrNull(cond);
  //     assert(condNew);
  //     conditionReg->insertOperands(0, condNew);     
  //   } else {
  //     auto *newOp = builder.clone(op, valueMap);
  //     for (size_t i = 0; i < newOp->getNumResults(); ++i) {
  //       auto newValue = newOp->getResult(i);
  //       auto oldValue = op.getResult(i);
  //       valueMap.map(oldValue, newValue);
  //     }
  //   }
  // }

  // builder.setInsertionPointToStart(&stgWhile.getScheduleBlock());

  // auto termConst = builder.create<arith::ConstantOp>(whileOp.getLoc(), builder.getIndexAttr(1));
  // auto term = stgWhile.getTerminator();
  // term.getIterArgsMutable().append(termConst.getResult());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp, func::ReturnOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  SmallVector<SmallVector<Operation*>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  // valueMap.clear();
  // for (size_t i = 0; i < iterArgs.size(); ++i)
  //   valueMap.map(whileOp.getAfter().getArgument(i),
  //                stgWhile.getScheduleBlock().getArgument(i));

  // Create the stages.
  Block &funcBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&funcBlock);

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
    auto isFuncTerminator = [funcOp](Operation *op) {
      return isa<func::ReturnOp>(op) && op->getParentOp() == funcOp;
    };
    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      for (auto *user : op->getUsers()) {
        if (opOrParentStartTime(problem, user) > startTime || isFuncTerminator(user)) {
          if (!opsWithReturns.contains(op)) {
            opsWithReturns.insert(op);
            stepTypes.append(op->getResultTypes().begin(),
                              op->getResultTypes().end());
          }
        }
      }
    }

    // Create the stage itself.
    auto stage =
        builder.create<STGStepOp>(stepTypes);
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

  // Update return with correct values
  auto *returnOp = funcOp.getBody().back().getTerminator();
  int numOperands = returnOp->getNumOperands();
  for (int i = 0; i < numOperands; ++i) {
    auto operand = returnOp->getOperand(i);
    auto newValue = valueMap.lookup(operand);
    returnOp->setOperand(i, newValue);
  }

  std::function<bool(Operation *)> inTopLevelStepOp = [&](Operation *op) {
    auto parent = op->getParentOfType<STGStepOp>();
    if (!parent)
      return false;

    if (isa<func::FuncOp>(parent->getParentOp()))
      return true;

    return inTopLevelStepOp(parent);
  };

  // Remove the loop nest from the IR.
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](Operation *op) {
    if ((isa<STGStepOp>(op) && isa<FuncOp>(op->getParentOp()))
      || inTopLevelStepOp(op)
      || isa<func::ReturnOp>(op))
      return;
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
