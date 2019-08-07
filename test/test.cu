
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <mathfu/vector.h>

using namespace mathfu;

int Factorial(int number ) {
  return number <= 1 ? number : Factorial(number-1) * number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
  REQUIRE( Factorial(0) == 1 );
  REQUIRE( Factorial(1) == 1 );
  REQUIRE( Factorial(2) == 2 );
  REQUIRE( Factorial(3) == 6 );
  REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE("Vectors", "[vector]") {
  Vector<int, 3> a = *new Vector<int, 3>(0, 4, 0);
  Vector<int, 3> b = *new Vector<int, 3>(0, 0, 3);
  int dist = a.Distance(a, b);
  REQUIRE(a == *new Vector<int, 3>(0, 4, 0));
  REQUIRE(b == *new Vector<int, 3>(0, 0, 3));
  REQUIRE(dist == 5);
}
