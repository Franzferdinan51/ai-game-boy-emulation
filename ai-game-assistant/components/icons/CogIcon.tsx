
import React from 'react';

const CogIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    {...props}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M9.594 3.94c.09-.542.56-1.007 1.11-1.226.554-.22 1.196-.22 1.75 0 .549.22 1.018.684 1.11 1.226M15.002 19.057a4.5 4.5 0 10-6.004 0M12 11.25a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 21a8.966 8.966 0 005.166-1.517M12 21a8.966 8.966 0 01-5.166-1.517m10.332 0A8.966 8.966 0 0012 3a8.966 8.966 0 00-5.166 16.517m10.332 0L12 11.25"
    />
  </svg>
);

export default CogIcon;
