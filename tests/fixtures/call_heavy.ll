; ModuleID = 'call_heavy'
; A function with many different call types: direct, intrinsic, and recursive.
; Expected: 12 instructions, 3 BBs, 0 loops,
;           direct_calls=2 (bar, baz), intrinsic_calls=2 (llvm.abs, llvm.smax),
;           self_recursive_calls=1 (call_heavy calls itself).

declare i32 @bar(i32)
declare i32 @baz(i32, i32)
declare i32 @llvm.abs.i32(i32, i1)
declare i32 @llvm.smax.i32(i32, i32)

define i32 @call_heavy(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else

then:
  %r1 = call i32 @bar(i32 %x)
  %r2 = call i32 @baz(i32 %x, i32 %r1)
  %r3 = call i32 @llvm.abs.i32(i32 %r2, i1 true)
  %r4 = call i32 @llvm.smax.i32(i32 %r3, i32 0)
  %r5 = call i32 @call_heavy(i32 %r4)
  ret i32 %r5

else:
  %neg = sub nsw i32 0, %x
  %r6 = call i32 @llvm.abs.i32(i32 %neg, i1 true)
  ret i32 %r6
}
