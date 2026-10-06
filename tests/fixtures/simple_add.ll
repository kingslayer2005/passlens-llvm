; ModuleID = 'simple_add'
; A trivial function: adds two integers. 
; Expected: 3 instructions (add, add, ret), 1 BB, 0 loops, 2 int_arith, 1 ret.

define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %sum = add nsw i32 %a, %b
  %result = add nsw i32 %sum, 1
  ret i32 %result
}
