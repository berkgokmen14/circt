#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;

/*
Pass coalesces loops with the following characteristics:
- There is a for loop with no iter args which includes one nested for loop with
one iter arg. The inner for loop doesn't have any for loops.
- Both loops have a step size of 1
- Both loops have constant bounds
*/
namespace {
struct HoistInterIterationArg
    : public circt::HoistInterIterationArgBase<HoistInterIterationArg> {
  using HoistInterIterationArgBase<
      HoistInterIterationArg>::HoistInterIterationArgBase;

  void runOnOperation() override {

    // Initialize builder and rewriter
    OpBuilder builder(getOperation().getBody());
    IRRewriter rewriter(builder);

    // Apply this pass for every for loop in the FuncOp
    for (auto forOp :
         llvm::SmallVector<AffineForOp>(getOperation().getOps<AffineForOp>())) {
      // Make sure there is only 1 for loop in forOp
      llvm::SmallVector<AffineForOp> childFors(forOp.getOps<AffineForOp>());
      if (childFors.size() == 1) {
        // Nested for loop in forOp
        auto childForOp = childFors.front();
        // Make sure there are no other for loops in the inner nested for loop
        if (childForOp.getBody()->getOps<LoopLikeOpInterface>().empty()) {
          // Check lower bound and upper bound for both loops are constant
          if (forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound() &&
              childForOp.hasConstantLowerBound() &&
              childForOp.hasConstantUpperBound()) {

            // Assert outer for loop and inner for loop have step size of 1
            assert(forOp.getStep() == 1 && childForOp.getStep() == 1);

            // Assert outer for loop has no iter operands and the inner loop has
            // only one iter operand to hoist
            assert(forOp.getNumIterOperands() == 0 &&
                   childForOp.getNumIterOperands() == 1);

            // Extract the upper bounds of the forOp and childForOp
            auto forBound = forOp.getConstantUpperBound();
            auto childForBound = childForOp.getConstantUpperBound();
            // Calculate new upper bound
            auto newForBound = forBound * childForBound;

            // Create coalesced for loop with new upper bound and copy child
            // iter operands
            builder.setInsertionPoint(forOp);
            auto newForOp = builder.create<AffineForOp>(
                forOp.getLoc(), 0, newForBound, 1, childForOp.getInits());

            builder.setInsertionPointToStart(
                &newForOp.getLoopRegions().front()->front());

            // Create a set that does dim % child for bound
            auto resetSet =
                IntegerSet::get(1, 0,
                                {builder.getAffineDimExpr(0) %
                                 builder.getAffineConstantExpr(childForBound)},
                                {true});

            // Create AffineIf where the if region represents the reset region,
            // and the then region represents the child loop region.
            // Additionally, output of the AffineIf is the gated iter arg
            auto affineIf = builder.create<AffineIfOp>(
                newForOp->getLoc(), builder.getI32Type(), resetSet,
                newForOp.getInductionVar(), true);
            auto &ifRegion = affineIf.getThenRegion();
            auto &elseRegion = affineIf.getElseRegion();

            // Replace childForOp induction variable with newForOp induction
            // variable.
            childForOp.getBody()->getArgument(0).replaceAllUsesWith(
                newForOp.getBody()->getArgument(0));
            // Replace childForOp iter argument with gatedIterArg.
            childForOp.getBody()->getArgument(1).replaceAllUsesWith(
                affineIf->getResult(0));

            // Erase all child block arguments since they are now unused
            childForOp.getBody()->eraseArguments(
                0, childForOp.getBody()->getArguments().size());

            // Replace uses of the child for loop with the iter operand of the
            // coalesced for loop
            childForOp->getResult(0).replaceAllUsesWith(
                newForOp.getBody()->getArgument(1));

            // Move child for loop body to else section
            rewriter.mergeBlocks(childForOp.getBody(), newForOp.getBody());

            // Replace all uses of the for op induction var with new for
            // op induction var
            forOp.getInductionVar().replaceAllUsesWith(
                newForOp.getInductionVar());

            // Erase for op block arguments for merging and forOp terminator if
            // present
            forOp.getBody()->eraseArgument(0);
            auto *forOpTerminator = forOp.getBody()->getTerminator();
            if (forOpTerminator)
              forOpTerminator->erase();

            // If an operation inside forOp is used in the childForOp, then move
            // that operation to outside the reset affineIf operation
            llvm::SmallVector<Operation *> forBodyOps;
            for (auto *region : forOp.getLoopRegions()) {
              for (auto &op : region->getOps()) {
                forBodyOps.push_back(&op);
              }
            }
            for (auto *op : forBodyOps) {
              op->dump();
              if (op->isUsedOutsideOfBlock(forOp.getBody())) {
                op->moveBefore(affineIf);
              }
            }

            // Move for loop body to if section
            rewriter.mergeBlocks(forOp.getBody(), &(ifRegion.front()));

            // Create yield reset for AffineIf
            mlir::ValueRange yieldOperands(childForOp.getInits());
            builder.setInsertionPointToEnd(&ifRegion.front());
            builder.create<AffineYieldOp>(newForOp.getLoc(), yieldOperands);
            // Create yield iter arg for AffineIf
            builder.setInsertionPointToEnd(&elseRegion.front());
            builder.create<AffineYieldOp>(newForOp.getLoc(),
                                          newForOp.getRegionIterArgs());

            // Erase the old forOp and childForOp
            forOp.erase();
            childForOp.erase();

            // Assert verification success
            assert(newForOp.verify().succeeded());
          }
        }
      }
    }
  }
};
} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createHoistInterIterationArgPass() {
  return std::make_unique<HoistInterIterationArg>();
}
} // namespace circt
