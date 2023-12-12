#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::affine;

namespace {
struct HoistInterIterationArg
    : public circt::HoistInterIterationArgBase<HoistInterIterationArg> {
  using HoistInterIterationArgBase<
      HoistInterIterationArg>::HoistInterIterationArgBase;

  void runOnOperation() override {
    llvm::errs() << "Printing step 0\n";
    OpBuilder builder(getOperation().getBody());

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
            // Assert outer for loop has no iter operands and the inner one has
            // at least one to hoist
            assert(forOp.getNumIterOperands() == 0 &&
                   childForOp.getNumIterOperands() >= 1);

            auto forBound = forOp.getConstantUpperBound();
            auto childForBound = childForOp.getConstantUpperBound();
            // Calculate new upper bound
            auto newForBound = forBound * childForBound;

            // Create coalesced for loop with nerForBound and
            // child iter operands
            builder.setInsertionPoint(forOp);
            auto newForOp =
                builder.create<AffineForOp>(forOp.getLoc(), 0, newForBound, 1,
                                            childForOp.getIterOperands());

            builder.setInsertionPointToStart(&newForOp.getLoopBody().front());
            // Create a set for division by 16
            auto resetSet =
                IntegerSet::get(1, 0,
                                {builder.getAffineDimExpr(0) %
                                 builder.getAffineConstantExpr(childForBound)},
                                {true});

            auto iterArgTypes = childForOp.getIterOperands().getTypes();
            // Now create the affineif
            auto affineIf = builder.create<AffineIfOp>(
                childForOp->getLoc(), iterArgTypes, resetSet,
                newForOp.getIterOperands(), true);

            auto &elseRegion = affineIf.getElseRegion();
            auto &ifRegion = affineIf.getThenRegion();

            IRRewriter rewriter(builder);

            // Iterate over child block arguments
            for (size_t i = 0; i < childForOp.getBody()->getArguments().size();
                 i++) {
              // Replace child arguments with the same order of new for op args
              childForOp.getBody()->getArgument(i).replaceAllUsesWith(
                  newForOp.getBody()->getArgument(i));
            }
            // Erase all child arguments since they are now unused
            childForOp.getBody()->eraseArguments(
                0, childForOp.getBody()->getArguments().size());

            // ASK: Replace all instances of the child for loop with the new for
            // loop
            childForOp->replaceAllUsesWith(affineIf);

            // Move child for loop body to else section
            rewriter.mergeBlocks(childForOp.getBody(), &(elseRegion.front()));

            // Replace all uses of the for op induction var with new for op
            // induction var
            forOp.getInductionVar().replaceAllUsesWith(
                newForOp.getInductionVar());

            // Erase for op block arguments for merging
            forOp.getBody()->eraseArgument(0);

            // Move for loop body to if section
            rewriter.mergeBlocks(forOp.getBody(), &(ifRegion.front()));

            // Create value range that includes 0
            mlir::ValueRange yieldOperands(childForOp.getIterOperands());

            rewriter.replaceOpWithNewOp<AffineYieldOp>(
                ifRegion.front().getTerminator(), yieldOperands);

            // Erase the old forOp and childForOp
            forOp.erase();
            childForOp.erase();

            newForOp->getParentOp()->dump();
            assert(newForOp.verify().succeeded());
            llvm::errs() << "Done dumping\n";
          }
        }
        // Now it's safe to call getNumOperands

        // auto forOpBody = forOp.getBody();
        // auto childForOpBody = childForOp.getBody();
        // auto forBound =
        //     (forOp.getUpperBound().getMap().getConstantResults().front());
        // auto upper = forBound * forBound;

        // auto newForOp =
        //     builder.create<AffineForOp>(getOperation().getLoc(), 0, upper,
        //     1);

        // auto &childForOpBlock = childForOp.getBody()->front();
        // auto newForOpBlock = newForOp.getBody();

        // OpBuilder::InsertionGuard guard(builder);
        // builder.setInsertionPointToStart(newForOpBlock);
        // builder.clone(childForOpBody->getOperations().front());

        // newForOp->dump();
        // llvm::errs() << "Done dumping\n";
        // exit(1);
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

/*
-2) Gather Perfect Loop Nests()
Collect all the loop nest then filter in next steps
-1) Innermost must return
0) Root for cannot have a yield. Make sure this is true
Make sure none of the loops don't have a result except for the innermost which
must have a result
1) Reconstruct loop nest immedately after
Use OPBuilder
2) Move inner body
region (can move entire body). can do it with remapping. Remapping (IRMap)
3) Insert Switch to replace old yield
4) Delete old nest

LLVM::ERRS() << "PRINTING STEP 1"
You can use fop->dump()
exit(1)


Add tablegen stuff in include/circt/transforms/passes.td


Might get false dependency
*/
